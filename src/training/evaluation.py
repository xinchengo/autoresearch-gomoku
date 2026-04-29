from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.algorithms.ppo import tensor_mask, tensor_obs
from src.training.checkpointing import load_torch_checkpoint
from src.training.config import TrainConfig
from src.gobang import GomokuEnv
from src.models import ActorCriticNet
from src.oracle import OracleAgent, make_oracle


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
    opponent: OracleAgent,
) -> int:
    env = GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row)
    obs, info = env.reset()
    done = False
    while not done:
        if info["current_player"] == model_player:
            action = model_action(model, obs, info["action_mask"], device)
        else:
            action = opponent.select_action(env.board, info["current_player"], info["action_mask"])
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return int(info["winner"])


def play_model_eval_game(
    model: ActorCriticNet,
    opponent_model: ActorCriticNet,
    cfg: TrainConfig,
    device: torch.device,
    model_player: int,
) -> int:
    env = GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row)
    obs, info = env.reset()
    done = False
    while not done:
        actor = model if info["current_player"] == model_player else opponent_model
        action = model_action(actor, obs, info["action_mask"], device)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return int(info["winner"])


def evaluate(model: ActorCriticNet, cfg: TrainConfig, device: torch.device) -> dict[str, float]:
    if cfg.eval_games <= 0:
        return {"score": 0.0, "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}

    opponent_metrics: dict[str, dict[str, float]] = {}
    for opponent_name in cfg.eval_opponents:
        opponent = make_oracle(opponent_name, cfg.board_size, cfg.n_in_row)
        opponent_metrics[opponent_name] = _evaluate_against(model, cfg, device, opponent)
    for spec in cfg.eval_checkpoints:
        name, checkpoint_path = _parse_checkpoint_spec(spec)
        opponent_model = load_checkpoint_model(checkpoint_path, cfg, device)
        opponent_metrics[name] = _evaluate_against_model(model, opponent_model, cfg, device)

    score = sum(metrics["score"] for metrics in opponent_metrics.values()) / len(opponent_metrics)
    win_rate = sum(metrics["win_rate"] for metrics in opponent_metrics.values()) / len(opponent_metrics)
    draw_rate = sum(metrics["draw_rate"] for metrics in opponent_metrics.values()) / len(opponent_metrics)
    loss_rate = sum(metrics["loss_rate"] for metrics in opponent_metrics.values()) / len(opponent_metrics)

    flat_metrics = {
        "score": score,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
    }
    for name, metrics in opponent_metrics.items():
        for key, value in metrics.items():
            flat_metrics[f"{name}_{key}"] = value
    return flat_metrics


def load_checkpoint_model(
    path: str | Path,
    cfg: TrainConfig,
    device: torch.device,
) -> ActorCriticNet:
    checkpoint = load_torch_checkpoint(path, device)
    saved_cfg = checkpoint.get("config", {})
    if int(saved_cfg.get("board_size", cfg.board_size)) != cfg.board_size:
        raise ValueError(f"checkpoint board_size does not match current config: {path}")
    if int(saved_cfg.get("n_in_row", cfg.n_in_row)) != cfg.n_in_row:
        raise ValueError(f"checkpoint n_in_row does not match current config: {path}")

    model = ActorCriticNet(
        board_size=cfg.board_size,
        channels=int(saved_cfg.get("channels", cfg.channels)),
        blocks=int(saved_cfg.get("blocks", cfg.blocks)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _evaluate_against(
    model: ActorCriticNet,
    cfg: TrainConfig,
    device: torch.device,
    opponent: OracleAgent,
) -> dict[str, float]:
    wins = draws = losses = 0
    for idx in range(cfg.eval_games):
        model_player = 1 if idx % 2 == 0 else -1
        winner = play_eval_game(model, cfg, device, model_player, opponent)
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


def _evaluate_against_model(
    model: ActorCriticNet,
    opponent_model: ActorCriticNet,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    wins = draws = losses = 0
    for idx in range(cfg.eval_games):
        model_player = 1 if idx % 2 == 0 else -1
        winner = play_model_eval_game(model, opponent_model, cfg, device, model_player)
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


def _parse_checkpoint_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = Path(path).stem
    clean_name = _metric_name(name)
    if not clean_name:
        raise ValueError(f"invalid checkpoint name in spec: {spec}")
    return f"ckpt_{clean_name}", path


def _metric_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip()).strip("_")
