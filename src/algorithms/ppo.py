from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.training.config import TrainConfig
from src.gobang import GomokuEnv
from src.models import ActorCriticNet
from src.oracle import make_oracle
from src.algorithms.numba_utils import compute_threat_bonus_numba


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


def _count_line(board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> int:
    size = int(board.shape[0])
    count = 0
    for sign in (-1, 1):
        r, c = int(row) + sign * dr, int(col) + sign * dc
        while 0 <= r < size and 0 <= c < size and int(board[r, c]) == player:
            count += 1
            r += sign * dr
            c += sign * dc
    return count


def compute_threat_bonus(
    board: np.ndarray, action: int, player: int, n_in_row: int, bonus_scale: float
) -> float:
    """Compute intermediate reward bonus for threat-related moves.

    Rewards blocking opponent threats and extending own patterns.
    This gives the model signal to learn basic Gomoku strategy from self-play.
    """
    if bonus_scale <= 0:
        return 0.0
    size = int(board.shape[0])
    row, col = divmod(int(action), size)
    opponent = -player
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    bonus = 0.0

    for dr, dc in directions:
        opp_count = _count_line(board, row, col, dr, dc, opponent)
        own_count = _count_line(board, row, col, dr, dc, player)

        if opp_count >= n_in_row - 1:
            bonus += bonus_scale * 0.3
        elif opp_count >= n_in_row - 2:
            bonus += bonus_scale * 0.1

        total_own = own_count + 1
        if total_own >= n_in_row - 1:
            bonus += bonus_scale * 0.2
        elif total_own >= n_in_row - 2:
            bonus += bonus_scale * 0.05

    return bonus


compute_threat_bonus = compute_threat_bonus_numba


def collect_vectorized_play(
    model: ActorCriticNet,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Tensor | int]:
    """Self-play with multiple parallel environments for batched GPU inference.

    Runs num_envs GomokuEnv instances in round-robin, batching observations
    into a single GPU forward pass. This increases GPU utilization by processing
    multiple environments per model.act() call.
    """
    num_envs = cfg.num_envs
    envs = [GomokuEnv(board_size=cfg.board_size, n_in_row=cfg.n_in_row) for _ in range(num_envs)]
    model.eval()

    # Per-environment state
    obs_list = [None] * num_envs
    mask_list = [None] * num_envs
    info_list = [None] * num_envs
    done_list = [True] * num_envs

    # Episode buffers per environment
    ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    ep_values: list[list[float]] = [[] for _ in range(num_envs)]

    # Global flat buffers
    obs_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    advantages: list[float] = []
    returns: list[float] = []
    total_games = 0

    while len(actions) < cfg.rollout_steps:
        # Reset completed environments
        for i in range(num_envs):
            if done_list[i]:
                obs, info = envs[i].reset()
                obs_list[i] = obs
                mask_list[i] = info["action_mask"].astype(np.bool_)
                info_list[i] = info
                done_list[i] = False

        # Collect active environments
        active = [i for i in range(num_envs) if not done_list[i]]
        if not active:
            continue

        # Batch observations and masks
        batch_obs = np.stack([obs_list[i] for i in active])
        batch_mask = np.stack([mask_list[i] for i in active])
        batch_tensor_obs = torch.as_tensor(batch_obs, dtype=torch.float32, device=device)
        batch_tensor_mask = torch.as_tensor(batch_mask, dtype=torch.bool, device=device)

        with torch.no_grad():
            actions_t, log_probs_t, values_t = model.act(
                batch_tensor_obs, batch_tensor_mask, deterministic=False
            )

        # Step each active environment
        for j, i in enumerate(active):
            action = int(actions_t[j].item())
            current_player = int(info_list[i]["current_player"])
            next_obs, reward, terminated, truncated, next_info = envs[i].step(action)
            bonus = compute_threat_bonus(
                envs[i].board, action, current_player, cfg.n_in_row, cfg.threat_bonus_scale
            )

            obs_rows.append(obs_list[i])
            mask_rows.append(mask_list[i])
            actions.append(action)
            old_log_probs.append(float(log_probs_t[j].item()))
            old_values.append(float(values_t[j].item()))
            ep_rewards[i].append(float(reward) + bonus)
            ep_values[i].append(float(values_t[j].item()))

            obs_list[i] = next_obs
            mask_list[i] = next_info["action_mask"].astype(np.bool_)
            info_list[i] = next_info
            done_list[i] = terminated or truncated

            if done_list[i]:
                if ep_rewards[i]:
                    ep_adv, ep_ret = compute_gae(
                        ep_rewards[i], ep_values[i], cfg.gamma, cfg.gae_lambda
                    )
                    advantages.extend(ep_adv)
                    returns.extend(ep_ret)
                ep_rewards[i] = []
                ep_values[i] = []
                total_games += 1

            if len(actions) >= cfg.rollout_steps:
                break

    # Flush incomplete episodes
    for i in range(num_envs):
        if ep_rewards[i]:
            ep_adv, ep_ret = compute_gae(
                ep_rewards[i], ep_values[i], cfg.gamma, cfg.gae_lambda
            )
            advantages.extend(ep_adv)
            returns.extend(ep_ret)

    n = len(obs_rows)
    return {
        "obs": torch.as_tensor(np.asarray(obs_rows[:n]), dtype=torch.float32, device=device),
        "masks": torch.as_tensor(np.asarray(mask_rows[:n]), dtype=torch.bool, device=device),
        "actions": torch.as_tensor(actions[:n], dtype=torch.long, device=device),
        "old_log_probs": torch.as_tensor(old_log_probs[:n], dtype=torch.float32, device=device),
        "old_values": torch.as_tensor(old_values[:n], dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(advantages[:n], dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns[:n], dtype=torch.float32, device=device),
        "steps": n,
        "games": total_games,
    }


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
            bonus = compute_threat_bonus(
                env.board, action, info["current_player"], cfg.n_in_row, cfg.threat_bonus_scale
            )

            obs_rows.append(obs)
            mask_rows.append(mask.astype(np.bool_))
            actions.append(action)
            old_log_probs.append(float(log_prob_t.item()))
            old_values.append(float(value_t.item()))
            ep_rewards.append(float(reward) + bonus)
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
