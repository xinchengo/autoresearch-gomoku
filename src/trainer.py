from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from tqdm import tqdm

from src.gobang import GomokuEnv
from src.models import ActorCriticNet


RESULTS_HEADER = "checkpoint\tsteps\tgames\tscore\twin_rate\tdraw_rate\tloss_rate\tdescription\n"


@dataclass
class TrainConfig:
    tag: str = "baseline"
    run_dir: str = "runs"
    board_size: int = 15
    n_in_row: int = 5
    seed: int = 1337
    device: str = "auto"
    duration_seconds: float = 3600.0
    checkpoint_interval_seconds: float = 1800.0
    eval_interval_seconds: float = 1800.0
    rollout_steps: int = 1024
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    clip_coef: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    channels: int = 64
    blocks: int = 4
    eval_games: int = 20
    description: str = "baseline PPO self-play"


@dataclass
class TrainState:
    env_steps: int = 0
    games: int = 0
    updates: int = 0
    best_score: float = -math.inf


def parse_duration(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    text = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", text)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid duration: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return amount * scale


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_obs(obs: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def tensor_mask(mask: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)


def choose_random_action(mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("no legal actions available")
    return int(np.random.choice(legal))


def compute_perspective_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] - gamma * running
        returns[idx] = running
    return returns


def collect_self_play(
    model: ActorCriticNet,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Tensor | int]:
    env = GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row)
    model.eval()

    obs_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    returns: list[float] = []
    total_games = 0

    while len(actions) < cfg.rollout_steps:
        obs, info = env.reset()
        ep_rewards: list[float] = []
        ep_start = len(actions)
        done = False

        while not done:
            mask = info["action_mask"]
            with torch.no_grad():
                action_t, log_prob_t, value_t = model.act(
                    tensor_obs(obs, device),
                    tensor_mask(mask, device),
                    deterministic=False,
                )
            action = int(action_t.item())
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            obs_rows.append(obs)
            mask_rows.append(mask.astype(np.bool_))
            actions.append(action)
            old_log_probs.append(float(log_prob_t.item()))
            old_values.append(float(value_t.item()))
            ep_rewards.append(float(reward))

            obs = next_obs
            info = next_info
            done = terminated or truncated

        returns.extend(compute_perspective_returns(ep_rewards, cfg.gamma))
        assert len(returns) == len(actions)
        assert ep_start < len(actions)
        total_games += 1

    return {
        "obs": torch.as_tensor(np.asarray(obs_rows), dtype=torch.float32, device=device),
        "masks": torch.as_tensor(np.asarray(mask_rows), dtype=torch.bool, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "old_log_probs": torch.as_tensor(old_log_probs, dtype=torch.float32, device=device),
        "old_values": torch.as_tensor(old_values, dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        "steps": len(actions),
        "games": total_games,
    }


def ppo_update(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor | int],
    cfg: TrainConfig,
) -> dict[str, float]:
    model.train()
    obs = batch["obs"]
    masks = batch["masks"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    old_values = batch["old_values"]
    returns = batch["returns"]
    assert isinstance(obs, Tensor)
    assert isinstance(masks, Tensor)
    assert isinstance(actions, Tensor)
    assert isinstance(old_log_probs, Tensor)
    assert isinstance(old_values, Tensor)
    assert isinstance(returns, Tensor)

    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = actions.shape[0]
    minibatch_size = min(cfg.minibatch_size, batch_size)

    stats: dict[str, list[float]] = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
    }

    for _ in range(cfg.ppo_epochs):
        indices = torch.randperm(batch_size, device=actions.device)
        for start in range(0, batch_size, minibatch_size):
            mb = indices[start : start + minibatch_size]
            new_log_probs, entropy, values = model.evaluate_actions(
                obs[mb], masks[mb], actions[mb]
            )
            log_ratio = new_log_probs - old_log_probs[mb]
            ratio = log_ratio.exp()
            unclipped = -advantages[mb] * ratio
            clipped = -advantages[mb] * torch.clamp(
                ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
            )
            policy_loss = torch.max(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[mb])
            entropy_loss = entropy.mean()
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            stats["policy_loss"].append(float(policy_loss.item()))
            stats["value_loss"].append(float(value_loss.item()))
            stats["entropy"].append(float(entropy_loss.item()))
            stats["approx_kl"].append(float(approx_kl.item()))

    return {key: float(np.mean(values)) for key, values in stats.items()}


@torch.no_grad()
def model_action(
    model: ActorCriticNet,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    deterministic: bool = True,
) -> int:
    model.eval()
    action, _log_prob, _value = model.act(
        tensor_obs(obs, device), tensor_mask(mask, device), deterministic=deterministic
    )
    return int(action.item())


def play_eval_game(
    model: ActorCriticNet,
    cfg: TrainConfig,
    device: torch.device,
    model_player: int,
) -> int:
    env = GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row)
    obs, info = env.reset()
    done = False
    while not done:
        if info["current_player"] == model_player:
            action = model_action(model, obs, info["action_mask"], device, deterministic=True)
        else:
            action = choose_random_action(info["action_mask"])
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return int(info["winner"])


def evaluate(model: ActorCriticNet, cfg: TrainConfig, device: torch.device) -> dict[str, float]:
    if cfg.eval_games <= 0:
        return {"score": 0.0, "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}

    wins = draws = losses = 0
    for idx in range(cfg.eval_games):
        model_player = 1 if idx % 2 == 0 else -1
        winner = play_eval_game(model, cfg, device, model_player)
        if winner == model_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    total = float(cfg.eval_games)
    win_rate = wins / total
    draw_rate = draws / total
    loss_rate = losses / total
    return {
        "score": win_rate + 0.5 * draw_rate,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
    }


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: TrainState,
) -> dict[str, Any]:
    return {
        "config": asdict(cfg),
        "state": asdict(state),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": rng_state(),
    }


def save_checkpoint(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: TrainState,
    run_path: Path,
    name: str,
) -> Path:
    ckpt_dir = run_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / name
    torch.save(checkpoint_payload(model, optimizer, cfg, state), path)
    latest = ckpt_dir / "latest.pt"
    torch.save(checkpoint_payload(model, optimizer, cfg, state), latest)
    return path


def load_torch_checkpoint(path: str | os.PathLike[str], device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def ensure_results_header(path: Path) -> None:
    if not path.exists():
        path.write_text(RESULTS_HEADER, encoding="utf-8")


def append_results(path: Path, checkpoint: str, state: TrainState, metrics: dict[str, float], description: str) -> None:
    ensure_results_header(path)
    row = (
        f"{checkpoint}\t{state.env_steps}\t{state.games}\t"
        f"{metrics['score']:.4f}\t{metrics['win_rate']:.4f}\t"
        f"{metrics['draw_rate']:.4f}\t{metrics['loss_rate']:.4f}\t{description}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def config_from_args(args: argparse.Namespace, saved: dict[str, Any] | None = None) -> TrainConfig:
    if saved:
        cfg = TrainConfig(**saved)
        return replace(
            cfg,
            tag=args.tag or cfg.tag,
            run_dir=args.run_dir,
            device=args.device,
            duration_seconds=args.duration,
            checkpoint_interval_seconds=args.checkpoint_interval,
            eval_interval_seconds=args.eval_interval,
            description=args.description or cfg.description,
        )
    return TrainConfig(
        tag=args.tag,
        run_dir=args.run_dir,
        board_size=args.board_size,
        n_in_row=args.n_in_row,
        seed=args.seed,
        device=args.device,
        duration_seconds=args.duration,
        checkpoint_interval_seconds=args.checkpoint_interval,
        eval_interval_seconds=args.eval_interval,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        channels=args.channels,
        blocks=args.blocks,
        eval_games=args.eval_games,
        description=args.description,
    )


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    saved_checkpoint = load_torch_checkpoint(args.resume, device) if args.resume else None
    cfg = config_from_args(args, saved_checkpoint.get("config") if saved_checkpoint else None)
    set_seed(cfg.seed)

    run_path = Path(cfg.run_dir) / cfg.tag
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

    if saved_checkpoint:
        model.load_state_dict(saved_checkpoint["model"])
        optimizer.load_state_dict(saved_checkpoint["optimizer"])
        state = TrainState(**saved_checkpoint.get("state", {}))
        restore_rng_state(saved_checkpoint.get("rng_state"))

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
            "resume": args.resume,
        },
    )

    progress = tqdm(total=cfg.duration_seconds, desc=f"train:{cfg.tag}", unit="s")
    last_progress = start_time
    try:
        while time.time() < end_time:
            batch = collect_self_play(model, cfg, device)
            update_stats = ppo_update(model, optimizer, batch, cfg)
            state.env_steps += int(batch["steps"])
            state.games += int(batch["games"])
            state.updates += 1

            now = time.time()
            remaining_progress = max(0.0, cfg.duration_seconds - progress.n)
            progress.update(min(max(0.0, now - last_progress), remaining_progress))
            last_progress = now

            record = {
                "event": "update",
                "time": now,
                "elapsed_seconds": now - start_time,
                "state": asdict(state),
                "rollout_steps": int(batch["steps"]),
                "rollout_games": int(batch["games"]),
                **update_stats,
            }
            append_jsonl(metrics_path, record)

            if now >= next_eval:
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
                next_eval = now + cfg.eval_interval_seconds

            if now >= next_checkpoint:
                save_checkpoint(
                    model,
                    optimizer,
                    cfg,
                    state,
                    run_path,
                    f"step_{state.env_steps}.pt",
                )
                next_checkpoint = now + cfg.checkpoint_interval_seconds
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

    print("---")
    print(f"score:             {final_metrics['score']:.4f}")
    print(f"win_rate:          {final_metrics['win_rate']:.4f}")
    print(f"draw_rate:         {final_metrics['draw_rate']:.4f}")
    print(f"loss_rate:         {final_metrics['loss_rate']:.4f}")
    print(f"env_steps:         {state.env_steps}")
    print(f"games:             {state.games}")
    print(f"updates:           {state.updates}")
    print(f"checkpoint:        {final_ckpt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO self-play trainer for Gomoku")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--n-in-row", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--duration", type=parse_duration, default=parse_duration("1h"))
    parser.add_argument("--checkpoint-interval", type=parse_duration, default=parse_duration("30m"))
    parser.add_argument("--eval-interval", type=parse_duration, default=parse_duration("30m"))
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--resume", default="")
    parser.add_argument("--description", default="baseline PPO self-play")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
