"""Supervised pretraining on threat scenarios.

Generates board positions where the correct move is to block/complete/extend
a threat. Trains the policy head using cross-entropy loss so the model learns
basic Gomoku strategy before PPO self-play fine-tuning.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gobang import GomokuEnv
from src.models import ActorCriticNet
from src.training.config import resolve_device

EMPTY = 0
BLACK = 1
WHITE = -1


def generate_scenario(
    board_size: int, n_in_row: int, scenario_type: str
) -> tuple[np.ndarray, int, int, int]:
    """Generate a threat scenario. Returns (board, correct_action, threat_player, model_player)."""
    board = np.zeros((board_size, board_size), dtype=np.int8)
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    if scenario_type == "block_open_four":
        player = WHITE
        model_player = BLACK
        dr, dc = random.choice(directions)
        length = n_in_row - 1
        placed = _place_line(board_size, dr, dc, length)
        if placed is None:
            return None
        start_r, start_c = placed[0]
        end_r, end_c = placed[-1]
        for r, c in placed:
            board[r, c] = player
        block_positions = [
            (start_r - dr, start_c - dc),
            (end_r + dr, end_c + dc),
        ]
        valid_blocks = [
            (r, c)
            for r, c in block_positions
            if 0 <= r < board_size and 0 <= c < board_size
        ]
        if not valid_blocks:
            return None
        block_r, block_c = random.choice(valid_blocks)
        correct_action = block_r * board_size + block_c
        return board, correct_action, player, model_player

    elif scenario_type == "complete_open_four":
        player = BLACK
        model_player = BLACK
        dr, dc = random.choice(directions)
        length = n_in_row - 1
        placed = _place_line(board_size, dr, dc, length)
        if placed is None:
            return None
        end_r, end_c = placed[-1]
        for r, c in placed:
            board[r, c] = player
        complete_positions = [(end_r + dr, end_c + dc)]
        valid_completes = [
            (r, c)
            for r, c in complete_positions
            if 0 <= r < board_size and 0 <= c < board_size
        ]
        if not valid_completes:
            return None
        cr, cc = valid_completes[0]
        correct_action = cr * board_size + cc
        return board, correct_action, player, model_player

    elif scenario_type == "block_open_three":
        player = WHITE
        model_player = BLACK
        dr, dc = random.choice(directions)
        length = n_in_row - 2
        placed = _place_line(board_size, dr, dc, length)
        if placed is None:
            return None
        end_r, end_c = placed[-1]
        for r, c in placed:
            board[r, c] = player
        block_positions = [(end_r + dr, end_c + dc)]
        valid_blocks = [
            (r, c)
            for r, c in block_positions
            if 0 <= r < board_size and 0 <= c < board_size
        ]
        if not valid_blocks:
            return None
        br, bc = valid_blocks[0]
        correct_action = br * board_size + bc
        return board, correct_action, player, model_player

    elif scenario_type == "extend_open_three":
        player = BLACK
        model_player = BLACK
        dr, dc = random.choice(directions)
        length = n_in_row - 2
        placed = _place_line(board_size, dr, dc, length)
        if placed is None:
            return None
        end_r, end_c = placed[-1]
        for r, c in placed:
            board[r, c] = player
        extend_positions = [(end_r + dr, end_c + dc)]
        valid_extends = [
            (r, c)
            for r, c in extend_positions
            if 0 <= r < board_size and 0 <= c < board_size
        ]
        if not valid_extends:
            return None
        er, ec = valid_extends[0]
        correct_action = er * board_size + ec
        return board, correct_action, player, model_player

    return None


def _place_line(
    board_size: int, dr: int, dc: int, length: int
) -> list[tuple[int, int]] | None:
    """Find a random valid placement for a line of given length."""
    attempts = 100
    for _ in range(attempts):
        r = random.randint(0, board_size - 1)
        c = random.randint(0, board_size - 1)
        positions = []
        cr, cc = r, c
        for _ in range(length):
            if not (0 <= cr < board_size and 0 <= cc < board_size):
                break
            positions.append((cr, cc))
            cr += dr
            cc += dc
        if len(positions) == length:
            return positions
    return None


def add_noise(board: np.ndarray, num_stones: int) -> None:
    """Add random stones to make the board more realistic."""
    size = board.shape[0]
    empty = [(r, c) for r in range(size) for c in range(size) if board[r, c] == EMPTY]
    random.shuffle(empty)
    for i in range(min(num_stones, len(empty))):
        r, c = empty[i]
        board[r, c] = BLACK if i % 2 == 0 else WHITE


def board_to_obs(board: np.ndarray, current_player: int) -> np.ndarray:
    """Convert board to 4-channel observation from current player's perspective."""
    own = board == current_player
    opponent = board == -current_player
    legal = board == EMPTY
    first_player = np.full_like(board, current_player == BLACK, dtype=np.float32)
    return np.stack(
        [own.astype(np.float32), opponent.astype(np.float32), legal.astype(np.float32), first_player],
        axis=0,
    )


def generate_dataset(
    num_samples: int, board_size: int = 12, n_in_row: int = 5
) -> tuple[list[np.ndarray], list[int]]:
    """Generate supervised dataset of (observation, correct_action) pairs."""
    scenarios = ["block_open_four", "complete_open_four", "block_open_three", "extend_open_three"]
    obs_list: list[np.ndarray] = []
    action_list: list[int] = []

    for _ in range(num_samples):
        stype = random.choice(scenarios)
        result = generate_scenario(board_size, n_in_row, stype)
        if result is None:
            continue
        board, correct_action, threat_player, model_player = result
        add_noise(board, random.randint(0, board_size // 2))
        obs = board_to_obs(board, model_player)
        obs_list.append(obs)
        action_list.append(correct_action)

    return obs_list, action_list


def pretrain(
    model: ActorCriticNet,
    obs_list: list[np.ndarray],
    action_list: list[int],
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> None:
    """Supervised pretraining of policy head using cross-entropy loss."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    obs_tensor = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(action_list, dtype=torch.long, device=device)
    N = len(obs_list)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        correct = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            logits, _value = model(obs_tensor[idx])
            loss = loss_fn(logits, action_tensor[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            correct += int((logits.argmax(dim=-1) == action_tensor[idx]).sum().item())

        acc = correct / N
        avg_loss = total_loss / N
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}: loss={avg_loss:.4f} acc={acc:.3f}")

    model.eval()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--board-size", type=int, default=12)
    p.add_argument("--n-in-row", type=int, default=5)
    p.add_argument("--output", default="runs/baseline-ladder-l0.pt")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    args = p.parse_args()

    device = resolve_device("auto")
    print(f"Device: {device}")
    print(f"Generating {args.samples} threat scenarios...")

    obs_list, action_list = generate_dataset(args.samples, args.board_size, args.n_in_row)
    print(f"Generated {len(obs_list)} valid scenarios")

    model = ActorCriticNet(
        board_size=args.board_size, channels=args.channels, blocks=args.blocks
    ).to(device)

    print(f"Pretraining ({args.epochs} epochs)...")
    pretrain(model, obs_list, action_list, device, epochs=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "board_size": args.board_size,
                "n_in_row": args.n_in_row,
                "channels": args.channels,
                "blocks": args.blocks,
            },
        },
        output_path,
    )
    print(f"Saved pretrained model to {output_path}")
