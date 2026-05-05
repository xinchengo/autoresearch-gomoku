from argparse import Namespace

import numpy as np
import torch

from src.models import ActorCriticNet
from src.algorithms.ppo import collect_self_play, ppo_update
from src.training.checkpointing import load_torch_checkpoint, save_checkpoint
from src.training.cli import build_parser
from src.training.config import TrainConfig, TrainState, parse_duration, resolve_device
from src.training.evaluation import evaluate
from src.training.runner import train


def test_parse_duration_units() -> None:
    assert parse_duration("30s") == 30
    assert parse_duration("2m") == 120
    assert parse_duration("1.5h") == 5400
    assert parse_duration("1d") == 86400


def test_parser_accepts_eval_opponents() -> None:
    parser = build_parser()
    args = parser.parse_args(["--eval-opponents", "random,tactical"])
    assert args.eval_opponents == ("random", "tactical")


def test_parser_accepts_eval_checkpoints() -> None:
    parser = build_parser()
    args = parser.parse_args(["--eval-checkpoints", "base=/tmp/base.pt,strong=/tmp/strong.pt"])
    assert args.eval_checkpoints == ("base=/tmp/base.pt", "strong=/tmp/strong.pt")


def test_model_masks_illegal_actions() -> None:
    model = ActorCriticNet(board_size=3, channels=8, blocks=1)
    obs = torch.zeros((1, 4, 3, 3), dtype=torch.float32)
    mask = torch.ones((1, 9), dtype=torch.bool)
    mask[0, 4] = False
    logits, value = model(obs, mask)
    assert logits.shape == (1, 9)
    assert value.shape == (1,)
    assert torch.isneginf(logits[0, 4]) or logits[0, 4].item() < -1e20


def test_collect_and_update_tiny_batch() -> None:
    cfg = TrainConfig(
        board_size=3,
        n_in_row=3,
        rollout_steps=8,
        channels=8,
        blocks=1,
        minibatch_size=8,
        ppo_epochs=1,
    )
    device = resolve_device("cpu")
    model = ActorCriticNet(board_size=3, channels=8, blocks=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = collect_self_play(model, cfg, device)
    assert int(batch["steps"]) >= 8
    assert int(batch["games"]) >= 1
    assert np.isfinite(batch["returns"].detach().cpu().numpy()).all()

    stats = ppo_update(model, optimizer, batch, cfg)
    assert stats["entropy"] > 0


def test_checkpoint_round_trip(tmp_path) -> None:
    cfg = TrainConfig(board_size=3, n_in_row=3, channels=8, blocks=1)
    state = TrainState(env_steps=12, games=2, updates=1, best_score=0.5)
    model = ActorCriticNet(board_size=3, channels=8, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    path = save_checkpoint(model, optimizer, cfg, state, tmp_path, "unit.pt")
    loaded = load_torch_checkpoint(path, torch.device("cpu"))
    assert loaded["state"]["env_steps"] == 12
    assert loaded["config"]["board_size"] == 3
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_evaluate_against_checkpoint_model(tmp_path) -> None:
    cfg = TrainConfig(
        board_size=3,
        n_in_row=3,
        channels=8,
        blocks=1,
        eval_games=2,
        eval_opponents=(),
    )
    state = TrainState()
    model = ActorCriticNet(board_size=3, channels=8, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(model, optimizer, cfg, state, tmp_path, "opponent.pt")

    eval_cfg = TrainConfig(
        board_size=3,
        n_in_row=3,
        channels=8,
        blocks=1,
        eval_games=2,
        eval_opponents=(),
        eval_checkpoints=(f"base={path}",),
    )
    metrics = evaluate(model, eval_cfg, torch.device("cpu"))
    assert "ckpt_base_score" in metrics
    assert "ckpt_base_win_rate" in metrics


def test_trainer_zero_duration_writes_final_checkpoint(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--tag",
            "unit",
            "--run-dir",
            str(tmp_path),
            "--board-size",
            "3",
            "--n-in-row",
            "3",
            "--duration",
            "0s",
            "--eval-games",
            "0",
            "--channels",
            "8",
            "--blocks",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert isinstance(args, Namespace)
    train(args)
    run_dirs = list(tmp_path.glob("*_unit"))
    assert len(run_dirs) == 1, f"expected 1 run dir, got {run_dirs}"
    run_dir = run_dirs[0]
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
