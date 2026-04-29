from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.net(x))


class ActorCriticNet(nn.Module):
    """Small pure neural policy-value network for Gomoku PPO."""

    def __init__(
        self,
        board_size: int = 15,
        in_channels: int = 4,
        channels: int = 64,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        self.board_size = int(board_size)
        self.num_actions = self.board_size * self.board_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * self.num_actions, self.num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.num_actions, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, obs: Tensor, action_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self.body(self.stem(obs))
        logits = self.policy_head(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, value = self(obs, action_mask)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: Tensor,
        action_mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, value = self(obs, action_mask)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

