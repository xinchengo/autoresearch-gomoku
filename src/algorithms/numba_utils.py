from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _count_line_numba(
    board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int
) -> int:
    size = board.shape[0]
    count = 0
    for sign in (-1, 1):
        r = row + sign * dr
        c = col + sign * dc
        while 0 <= r < size and 0 <= c < size and board[r, c] == player:
            count += 1
            r += sign * dr
            c += sign * dc
    return count


@njit(cache=True)
def compute_threat_bonus_numba(
    board: np.ndarray, action: int, player: int, n_in_row: int, bonus_scale: float
) -> float:
    if bonus_scale <= 0.0:
        return 0.0
    size = board.shape[0]
    row = action // size
    col = action % size
    opponent = -player
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    bonus = 0.0

    for dr, dc in directions:
        opp_count = _count_line_numba(board, row, col, dr, dc, opponent)
        own_count = _count_line_numba(board, row, col, dr, dc, player)

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
