from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.training.config import TrainConfig, TrainState
from src.models import ActorCriticNet


RESULTS_HEADER = (
    "checkpoint\tsteps\tgames\tscore\twin_rate\tdraw_rate\tloss_rate\tdetails_json\tdescription\n"
)


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
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except (TypeError, ValueError, RuntimeError):
        pass


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
    payload = checkpoint_payload(model, optimizer, cfg, state)
    torch.save(payload, path)
    torch.save(payload, ckpt_dir / "latest.pt")
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


def append_results(
    path: Path,
    checkpoint: str,
    state: TrainState,
    metrics: dict[str, float],
    description: str,
) -> None:
    ensure_results_header(path)
    row = (
        f"{checkpoint}\t{state.env_steps}\t{state.games}\t"
        f"{metrics['score']:.4f}\t{metrics['win_rate']:.4f}\t"
        f"{metrics['draw_rate']:.4f}\t{metrics['loss_rate']:.4f}\t"
        f"{json.dumps(metrics, sort_keys=True, separators=(',', ':'))}\t{description}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)
