import numpy as np
import pytest

from src.gobang.env import BLACK, EMPTY, WHITE, GomokuEnv, check_winner


def test_horizontal_win_from_env_step() -> None:
    env = GomokuEnv(board_size=3, n_in_row=3)
    _obs, info = env.reset()
    assert info["current_player"] == BLACK

    for action in [0, 3, 1, 4]:
        _obs, reward, terminated, _truncated, _info = env.step(action)
        assert reward == 0.0
        assert not terminated

    _obs, reward, terminated, _truncated, info = env.step(2)
    assert terminated
    assert reward == 1.0
    assert info["winner"] == BLACK


@pytest.mark.parametrize(
    ("stones", "winner"),
    [
        ([(0, 0), (1, 0), (2, 0)], BLACK),
        ([(0, 0), (1, 1), (2, 2)], BLACK),
        ([(0, 2), (1, 1), (2, 0)], BLACK),
    ],
)
def test_check_winner_directions(stones: list[tuple[int, int]], winner: int) -> None:
    board = np.zeros((3, 3), dtype=np.int8)
    for row, col in stones:
        board[row, col] = winner
    assert check_winner(board, 3) == winner


def test_draw_detection() -> None:
    env = GomokuEnv(board_size=3, n_in_row=3)
    env.reset()
    for action in [0, 1, 2, 4, 3, 5, 7, 6]:
        _obs, _reward, terminated, _truncated, _info = env.step(action)
        assert not terminated

    _obs, reward, terminated, _truncated, info = env.step(8)
    assert terminated
    assert reward == 0.0
    assert info["winner"] == EMPTY


def test_illegal_move_ends_game_for_opponent() -> None:
    env = GomokuEnv(board_size=3, n_in_row=3)
    env.reset()
    env.step(0)
    _obs, reward, terminated, _truncated, info = env.step(0)
    assert terminated
    assert reward == -1.0
    assert info["winner"] == BLACK
    assert info["illegal_action"] == 0


def test_observation_and_mask_are_current_player_perspective() -> None:
    env = GomokuEnv(board_size=3, n_in_row=3)
    obs, info = env.reset()
    assert obs.shape == (4, 3, 3)
    assert info["action_mask"].sum() == 9

    obs, _reward, _terminated, _truncated, info = env.step(0)
    assert info["current_player"] == WHITE
    assert obs[1, 0, 0] == 1.0
    assert not info["action_mask"][0]

