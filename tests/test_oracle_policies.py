import numpy as np

from src.oracle import RandomPolicy, TacticalPolicy


def test_random_policy_returns_legal_action() -> None:
    board = np.zeros((3, 3), dtype=np.int8)
    legal_mask = np.array([False, True, False, False, False, False, False, False, False])
    policy = RandomPolicy()
    assert policy.select_action(board, 1, legal_mask) == 1


def test_tactical_policy_blocks_opponent_open_three() -> None:
    board = np.zeros((5, 5), dtype=np.int8)
    board[2, 1:4] = -1
    legal_mask = board.reshape(-1) == 0
    policy = TacticalPolicy(board_size=5, n_in_row=5)

    action = policy.select_action(board, 1, legal_mask)
    assert action in {2 * 5 + 0, 2 * 5 + 4}


def test_tactical_policy_extends_own_open_three() -> None:
    board = np.zeros((5, 5), dtype=np.int8)
    board[2, 1:4] = 1
    legal_mask = board.reshape(-1) == 0
    policy = TacticalPolicy(board_size=5, n_in_row=5)

    action = policy.select_action(board, 1, legal_mask)
    assert action in {2 * 5 + 0, 2 * 5 + 4}


def test_tactical_policy_prefers_center_when_board_is_empty() -> None:
    np.random.seed(0)
    board = np.zeros((5, 5), dtype=np.int8)
    legal_mask = board.reshape(-1) == 0
    policy = TacticalPolicy(board_size=5, n_in_row=5)

    action = policy.select_action(board, 1, legal_mask)
    assert 0 <= action < 25
    assert legal_mask[action]

