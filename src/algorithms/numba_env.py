from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def step_board_numba(
    board: np.ndarray,
    action: int,
    current_player: int,
    n_in_row: int,
    illegal_move_ends_game: bool = True,
) -> tuple[np.ndarray, int, float, bool, np.ndarray]:
    size = board.shape[0]
    row = action // size
    col = action % size

    if row < 0 or row >= size or col < 0 or col >= size:
        if illegal_move_ends_game:
            return board, -current_player, -1.0, True, action_mask_numba(board)
        return board, current_player, -1.0, False, action_mask_numba(board)

    if board[row, col] != 0:
        if illegal_move_ends_game:
            return board, -current_player, -1.0, True, action_mask_numba(board)
        return board, current_player, -1.0, False, action_mask_numba(board)

    board[row, col] = current_player
    new_player = -current_player
    reward = 0.0
    terminated = False

    winner = _check_winner_numba(board, n_in_row, row, col)
    if winner != 0:
        reward = 1.0
        terminated = True
        new_player = current_player
    elif _is_full(board):
        terminated = True
        new_player = current_player

    mask = action_mask_numba(board)
    return board, new_player, reward, terminated, mask


@njit(cache=True, nogil=True)
def _check_winner_numba(board: np.ndarray, n_in_row: int, last_row: int, last_col: int) -> int:
    size = board.shape[0]
    player = board[last_row, last_col]
    if player == 0:
        return 0
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for dr, dc in directions:
        count = 1
        for sign in (-1, 1):
            rr = last_row + sign * dr
            cc = last_col + sign * dc
            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
        if count >= n_in_row:
            return player
    return 0


@njit(cache=True, nogil=True)
def _is_full(board: np.ndarray) -> bool:
    for i in range(board.size):
        if board.flat[i] == 0:
            return False
    return True


@njit(cache=True, nogil=True)
def action_mask_numba(board: np.ndarray) -> np.ndarray:
    size = board.shape[0]
    mask = np.empty(size * size, dtype=np.bool_)
    for i in range(size * size):
        mask[i] = (board.flat[i] == 0)
    return mask


@njit(cache=True, nogil=True)
def observation_numba(board: np.ndarray, current_player: int) -> np.ndarray:
    size = board.shape[0]
    obs = np.empty((4, size, size), dtype=np.float32)
    is_black = (current_player == 1)
    for r in range(size):
        for c in range(size):
            v = board[r, c]
            obs[0, r, c] = 1.0 if v == current_player else 0.0
            obs[1, r, c] = 1.0 if v == -current_player else 0.0
            obs[2, r, c] = 1.0 if v == 0 else 0.0
            obs[3, r, c] = 1.0 if is_black else 0.0
    return obs


class FastEnvState:
    __slots__ = ("board", "current_player", "move_count")

    def __init__(self, board_size: int, current_player: int = 1):
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = current_player
        self.move_count = 0

    def reset(self) -> tuple[np.ndarray, np.ndarray, int]:
        self.board.fill(0)
        self.current_player = 1
        self.move_count = 0
        obs = observation_numba(self.board, self.current_player)
        mask = action_mask_numba(self.board)
        return obs, mask, self.current_player

    def step(self, action: int, n_in_row: int) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
        board, new_player, reward, terminated, mask = step_board_numba(
            self.board, action, self.current_player, n_in_row
        )
        self.board = board
        self.current_player = new_player
        self.move_count += 1
        obs = observation_numba(self.board, self.current_player)
        return obs, mask, reward, terminated, self.current_player
