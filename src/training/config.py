from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import numpy as np
import torch


@dataclass
class TrainConfig:
    tag: str = "baseline"
    run_dir: str = "runs"
    board_size: int = 12
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
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    threat_bonus_scale: float = 0.3
    freeze_body: bool = False
    mixed_opponents: bool = True
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    lr_min_factor: float = 0.1
    channels: int = 64
    blocks: int = 4
    eval_games: int = 40
    num_envs: int = 32
    vectorized_collect: bool = True
    grad_accum_steps: int = 4
    eval_opponents: tuple[str, ...] = ("random", "tactical")
    eval_checkpoints: tuple[str, ...] = ()
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
    return amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def parse_opponents(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    opponents = tuple(part.strip() for part in value.split(",") if part.strip())
    if not opponents:
        raise argparse.ArgumentTypeError("at least one eval opponent is required")
    return opponents


def parse_checkpoint_specs(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    specs = tuple(part.strip() for part in value.split(",") if part.strip())
    if not specs:
        raise argparse.ArgumentTypeError("at least one checkpoint spec is required")
    return specs


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


def load_config_preset(name: str) -> TrainConfig:
    module = import_module(f"src.configs.{name}")
    if hasattr(module, "get_config"):
        cfg = module.get_config()
        if not isinstance(cfg, TrainConfig):
            raise TypeError(f"src.configs.{name}.get_config() must return TrainConfig")
        return cfg
    if hasattr(module, "CONFIG"):
        data = module.CONFIG
        if isinstance(data, TrainConfig):
            return data
        if isinstance(data, dict):
            return TrainConfig(**data)
    raise AttributeError(f"src.configs.{name} must define CONFIG or get_config()")


def config_from_args(
    args: argparse.Namespace,
    saved: dict[str, Any] | None = None,
) -> TrainConfig:
    if saved:
        cfg = TrainConfig(**saved)
    else:
        cfg = load_config_preset(args.config)

    overrides = {
        "tag": args.tag,
        "run_dir": args.run_dir,
        "board_size": args.board_size,
        "n_in_row": args.n_in_row,
        "seed": args.seed,
        "device": args.device,
        "duration_seconds": args.duration,
        "checkpoint_interval_seconds": args.checkpoint_interval,
        "eval_interval_seconds": args.eval_interval,
        "rollout_steps": args.rollout_steps,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.minibatch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_coef": args.clip_coef,
        "value_clip_coef": args.value_clip_coef,
        "threat_bonus_scale": args.threat_bonus_scale,
        "entropy_coef": args.entropy_coef,
        "value_coef": args.value_coef,
        "max_grad_norm": args.max_grad_norm,
        "learning_rate": args.learning_rate,
        "channels": args.channels,
        "blocks": args.blocks,
        "num_envs": args.num_envs,
        "vectorized_collect": args.vectorized_collect,
        "eval_games": args.eval_games,
        "eval_opponents": args.eval_opponents,
        "eval_checkpoints": args.eval_checkpoints,
        "description": args.description,
        "freeze_body": args.freeze_body,
    }
    return replace(cfg, **{key: value for key, value in overrides.items() if value is not None})
