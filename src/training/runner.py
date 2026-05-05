from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.algorithms.ppo import collect_self_play, collect_vectorized_play, ppo_update
from src.training.checkpointing import (
    append_jsonl,
    append_results,
    ensure_results_header,
    load_torch_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from src.training.config import TrainConfig, TrainState, config_from_args, resolve_device, set_seed
from src.training.evaluation import evaluate
from src.models import ActorCriticNet


def train(args) -> None:
    load_device = resolve_device(args.device or "auto")
    saved_checkpoint = load_torch_checkpoint(args.resume, load_device) if args.resume else None
    pretrained_checkpoint = load_torch_checkpoint(args.pretrained, load_device) if args.pretrained else None
    cfg = config_from_args(args, saved_checkpoint.get("config") if saved_checkpoint else None)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    run_path = Path(cfg.run_dir) / cfg.tag
    run_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_path = Path(cfg.run_dir) / f"{timestamp}_{cfg.tag}"
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_path = run_path / "metrics.jsonl"
    results_path = run_path / "results.tsv"
    ensure_results_header(results_path)

    model = ActorCriticNet(
        board_size=cfg.board_size,
        channels=cfg.channels,
        blocks=cfg.blocks,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    state = TrainState()

    if pretrained_checkpoint:
        model.load_state_dict(pretrained_checkpoint["model"])
        if cfg.freeze_body:
            for name, param in model.named_parameters():
                if "stem" in name or "body" in name:
                    param.requires_grad = False
    elif saved_checkpoint:
        model.load_state_dict(saved_checkpoint["model"])
        optimizer.load_state_dict(saved_checkpoint["optimizer"])
        state = TrainState(**saved_checkpoint.get("state", {}))
        restore_rng_state(saved_checkpoint.get("rng_state"))

    _run_training_loop(model, optimizer, cfg, state, run_path, metrics_path, results_path, device, args.resume)


def _run_training_loop(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: TrainState,
    run_path: Path,
    metrics_path: Path,
    results_path: Path,
    device: torch.device,
    resume_path: str,
) -> None:
    start_time = time.time()
    end_time = start_time + cfg.duration_seconds
    next_checkpoint = start_time + cfg.checkpoint_interval_seconds
    next_eval = start_time

    append_jsonl(
        metrics_path,
        {
            "event": "start",
            "time": start_time,
            "config": asdict(cfg),
            "state": asdict(state),
            "resume": resume_path,
        },
    )

    progress = tqdm(total=cfg.duration_seconds, desc=f"train:{cfg.tag}", unit="s")
    last_progress = start_time
    try:
        while time.time() < end_time:
            if cfg.vectorized_collect:
                batch = collect_vectorized_play(model, cfg, device)
            else:
                batch = collect_self_play(model, cfg, device)
            update_stats = ppo_update(model, optimizer, batch, cfg)
            state.env_steps += int(batch["steps"])
            state.games += int(batch["games"])
            state.updates += 1

            now = time.time()
            elapsed = now - start_time
            remaining_progress = max(0.0, cfg.duration_seconds - progress.n)
            progress.update(min(max(0.0, now - last_progress), remaining_progress))
            last_progress = now

            append_jsonl(
                metrics_path,
                {
                    "event": "update",
                    "time": now,
                    "elapsed_seconds": elapsed,
                    "state": asdict(state),
                    "rollout_steps": int(batch["steps"]),
                    "rollout_games": int(batch["games"]),
                    "updates_per_hour": state.updates / max(elapsed, 1) * 3600,
                    "steps_per_second": state.env_steps / max(elapsed, 1),
                    **update_stats,
                },
            )
            next_eval = _maybe_evaluate(
                model, optimizer, cfg, state, run_path, metrics_path, results_path, device, start_time, now, next_eval
            )
            next_checkpoint = _maybe_checkpoint(
                model, optimizer, cfg, state, run_path, now, next_checkpoint
            )
    finally:
        progress.close()

    final_metrics = evaluate(model, cfg, device)
    final_ckpt = save_checkpoint(model, optimizer, cfg, state, run_path, "final.pt")
    append_jsonl(
        metrics_path,
        {
            "event": "final",
            "time": time.time(),
            "elapsed_seconds": time.time() - start_time,
            "state": asdict(state),
            "checkpoint": str(final_ckpt),
            **final_metrics,
        },
    )
    append_results(results_path, "final.pt", state, final_metrics, cfg.description)
    _print_summary(final_metrics, state, final_ckpt)


def _maybe_evaluate(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: TrainState,
    run_path: Path,
    metrics_path: Path,
    results_path: Path,
    device: torch.device,
    start_time: float,
    now: float,
    next_eval: float,
) -> float:
    if now < next_eval:
        return next_eval

    metrics = evaluate(model, cfg, device)
    is_best = metrics["score"] > state.best_score
    if is_best:
        state.best_score = metrics["score"]
    checkpoint_name = f"step_{state.env_steps}.pt"
    ckpt_path = save_checkpoint(model, optimizer, cfg, state, run_path, checkpoint_name)
    if is_best:
        save_checkpoint(model, optimizer, cfg, state, run_path, "best.pt")

    append_jsonl(
        metrics_path,
        {
            "event": "eval",
            "time": time.time(),
            "elapsed_seconds": time.time() - start_time,
            "state": asdict(state),
            "checkpoint": str(ckpt_path),
            **metrics,
        },
    )
    append_results(results_path, checkpoint_name, state, metrics, cfg.description)
    return now + cfg.eval_interval_seconds


def _maybe_checkpoint(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: TrainState,
    run_path: Path,
    now: float,
    next_checkpoint: float,
) -> float:
    if now < next_checkpoint:
        return next_checkpoint
    save_checkpoint(model, optimizer, cfg, state, run_path, f"step_{state.env_steps}.pt")
    return now + cfg.checkpoint_interval_seconds


def _print_summary(metrics: dict[str, float], state: TrainState, checkpoint: Path) -> None:
    print("---")
    print(f"score:             {metrics['score']:.4f}")
    print(f"win_rate:          {metrics['win_rate']:.4f}")
    print(f"draw_rate:         {metrics['draw_rate']:.4f}")
    print(f"loss_rate:         {metrics['loss_rate']:.4f}")
    print(f"env_steps:         {state.env_steps}")
    print(f"games:             {state.games}")
    print(f"updates:           {state.updates}")
    print(f"checkpoint:        {checkpoint}")
