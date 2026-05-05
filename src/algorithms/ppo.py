from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.training.config import TrainConfig
from src.gobang import GomokuEnv
from src.models import ActorCriticNet


def tensor_obs(obs: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def tensor_mask(mask: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)


def compute_perspective_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] - gamma * running
        returns[idx] = running
    return returns


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute GAE advantages and returns for a single episode.

    Uses perspective-aware GAE for zero-sum games.
    The model's value prediction is always from the current player's perspective,
    so V(s_{t+1}) is from the opponent's view. The advantage accumulation uses
    subtraction (not addition) to flip the opponent's advantage sign.
    """
    T = len(rewards)
    advantages = [0.0 for _ in range(T)]
    returns = [0.0 for _ in range(T)]
    gae = 0.0
    for t in range(T - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] - gamma * next_value - values[t]
        gae = delta - gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns


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
    advantages: list[float] = []
    returns: list[float] = []
    total_games = 0

    while len(actions) < cfg.rollout_steps:
        obs, info = env.reset()
        ep_rewards: list[float] = []
        ep_values: list[float] = []
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
            ep_values.append(float(value_t.item()))

            obs = next_obs
            info = next_info
            done = terminated or truncated

        ep_adv, ep_ret = compute_gae(ep_rewards, ep_values, cfg.gamma, cfg.gae_lambda)
        advantages.extend(ep_adv)
        returns.extend(ep_ret)
        assert len(returns) == len(actions)
        total_games += 1

    return {
        "obs": torch.as_tensor(np.asarray(obs_rows), dtype=torch.float32, device=device),
        "masks": torch.as_tensor(np.asarray(mask_rows), dtype=torch.bool, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "old_log_probs": torch.as_tensor(old_log_probs, dtype=torch.float32, device=device),
        "old_values": torch.as_tensor(old_values, dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
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
    obs = _require_tensor(batch["obs"])
    masks = _require_tensor(batch["masks"])
    actions = _require_tensor(batch["actions"])
    old_log_probs = _require_tensor(batch["old_log_probs"])
    old_values = _require_tensor(batch["old_values"])
    returns = _require_tensor(batch["returns"])
    advantages = _require_tensor(batch["advantages"])
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
            policy_loss = torch.max(
                -advantages[mb] * ratio,
                -advantages[mb]
                * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
            ).mean()
            value_pred_clipped = old_values[mb] + torch.clamp(
                values - old_values[mb], -cfg.value_clip_coef, cfg.value_clip_coef
            )
            value_loss = torch.max(
                F.mse_loss(values, returns[mb]),
                F.mse_loss(value_pred_clipped, returns[mb]),
            )
            entropy_loss = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

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


def _require_tensor(value: Tensor | int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"expected Tensor, got {type(value).__name__}")
    return value
