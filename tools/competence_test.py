"""Basic Gomoku competence tests for trained models.

Tests whether a model can perform elementary Gomoku operations:
- Block an opponent's open four
- Complete own open four
- Block an opponent's open three
- Extend own open three
- Avoid illegal moves

These are the minimum competence bar for any useful Gomoku agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gobang import GomokuEnv
from src.training.evaluation import model_action
from src.training.checkpointing import load_torch_checkpoint
from src.training.config import TrainConfig, resolve_device
from src.models import ActorCriticNet

EMPTY = 0
BLACK = 1
WHITE = -1


def _place(board: np.ndarray, positions: list[tuple[int, int]], player: int) -> None:
    for r, c in positions:
        board[r, c] = player


def _model_act(model, env, device) -> tuple[int, int, int]:
    obs = env._observation()
    mask = env.action_mask()
    action = model_action(model, obs, mask, device, deterministic=True)
    row, col = divmod(action, env.board_size)
    return action, row, col


def test_block_open_four(model, cfg, device, board_size, n_in_row):
    """WHITE has open four at (0,0)-(0,3). Model (BLACK) must block at (0,4)."""
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
    _place(env.board, [(0, 0), (0, 1), (0, 2), (0, 3)], WHITE)
    env.current_player = BLACK
    action, row, col = _model_act(model, env, device)
    passed = (row == 0 and col == 4)
    return {"test": "block_open_four", "passed": passed, "action": (row, col), "expected": (0, 4)}


def test_complete_open_four(model, cfg, device, board_size, n_in_row):
    """BLACK has open four at (0,0)-(0,3). Must complete at (0,4) to win."""
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
    _place(env.board, [(0, 0), (0, 1), (0, 2), (0, 3)], BLACK)
    env.current_player = BLACK
    action, row, col = _model_act(model, env, device)
    passed = (row == 0 and col == 4)
    return {"test": "complete_open_four", "passed": passed, "action": (row, col), "expected": (0, 4)}


def test_block_open_three(model, cfg, device, board_size, n_in_row):
    """WHITE has open three at (0,0)-(0,2). Model must block at (0,3)."""
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
    _place(env.board, [(0, 0), (0, 1), (0, 2)], WHITE)
    env.current_player = BLACK
    action, row, col = _model_act(model, env, device)
    passed = (row == 0 and col == 3)
    return {"test": "block_open_three", "passed": passed, "action": (row, col), "expected": (0, 3)}


def test_extend_open_three(model, cfg, device, board_size, n_in_row):
    """BLACK has open three at (0,0)-(0,2). Must extend to (0,3)."""
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
    _place(env.board, [(0, 0), (0, 1), (0, 2)], BLACK)
    env.current_player = BLACK
    action, row, col = _model_act(model, env, device)
    passed = (row == 0 and col == 3)
    return {"test": "extend_open_three", "passed": passed, "action": (row, col), "expected": (0, 3)}


def test_avoid_illegal(model, cfg, device, board_size, n_in_row):
    """Fill board except (5,5). Model must pick that cell."""
    env = GomokuEnv(board_size=board_size, n_in_row=n_in_row)
    for r in range(board_size):
        for c in range(board_size):
            if (r, c) != (5, 5):
                env.board[r, c] = BLACK if (r + c) % 2 == 0 else WHITE
    env.current_player = BLACK
    action, row, col = _model_act(model, env, device)
    passed = (row == 5 and col == 5)
    return {"test": "avoid_illegal", "passed": passed, "action": (row, col), "expected": (5, 5)}


ALL_TESTS = [
    test_block_open_four,
    test_complete_open_four,
    test_block_open_three,
    test_extend_open_three,
    test_avoid_illegal,
]


def run_competence_suite(checkpoint_path, device="auto"):
    dev = resolve_device(device)
    checkpoint = load_torch_checkpoint(checkpoint_path, dev)
    saved_cfg = checkpoint.get("config", {})
    board_size = int(saved_cfg.get("board_size", 12))
    n_in_row = int(saved_cfg.get("n_in_row", 5))

    model = ActorCriticNet(
        board_size=board_size,
        channels=int(saved_cfg.get("channels", 64)),
        blocks=int(saved_cfg.get("blocks", 4)),
    ).to(dev)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cfg = TrainConfig(board_size=board_size, n_in_row=n_in_row)

    results = {}
    for test_fn in ALL_TESTS:
        r = test_fn(model, cfg, dev, board_size, n_in_row)
        results[r["test"]] = r["passed"]
        status = "PASS" if r["passed"] else "FAIL"
        a_row, a_col = r["action"]
        e_row, e_col = r["expected"]
        print(f"  [{status}] {r['test']}: played ({a_row},{a_col}), expected ({e_row},{e_col})")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to checkpoint .pt file")
    args = p.parse_args()

    print(f"Competence test: {args.checkpoint}")
    print("-" * 50)

    results = run_competence_suite(args.checkpoint)

    passed = sum(results.values())
    total = len(results)
    print("-" * 50)
    print(f"Result: {passed}/{total} passed")
    if passed < total:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
