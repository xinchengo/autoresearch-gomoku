from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TacticalPolicy:
    """Rule baseline with immediate threat blocking and center-biased shape growth."""

    board_size: int
    n_in_row: int
    name: str = "tactical"

    def select_action(self, board: np.ndarray, player: int, legal_mask: np.ndarray) -> int:
        opponent = -int(player)
        for target in (opponent, int(player)):
            candidates = [
                action
                for action in np.flatnonzero(legal_mask)
                if self._creates_priority_shape(board, int(action), target)
            ]
            if candidates:
                return self._sample_by_center_distance(candidates)

        best_actions = self._longest_segment_actions(board, int(player), legal_mask)
        return self._sample_by_center_distance(best_actions)

    def _creates_priority_shape(self, board: np.ndarray, action: int, player: int) -> bool:
        row, col = divmod(action, self.board_size)
        if board[row, col] != 0:
            return False

        test_board = board.copy()
        test_board[row, col] = player
        for dr, dc in _DIRECTIONS:
            length, open_ends = _line_shape(test_board, row, col, dr, dc, player)
            if length >= self.n_in_row:
                return True
            if length >= self.n_in_row - 1 and open_ends >= 1:
                return True
            if length >= self.n_in_row - 2 and open_ends == 2:
                return True
        return False

    def _longest_segment_actions(
        self,
        board: np.ndarray,
        player: int,
        legal_mask: np.ndarray,
    ) -> list[int]:
        scored: list[tuple[int, int, int]] = []
        for action in np.flatnonzero(legal_mask):
            row, col = divmod(int(action), self.board_size)
            test_board = board.copy()
            test_board[row, col] = player
            lengths = [
                _line_shape(test_board, row, col, dr, dc, player)[0]
                for dr, dc in _DIRECTIONS
            ]
            longest = max(lengths)
            count = sum(length == longest for length in lengths)
            scored.append((longest, count, int(action)))

        if not scored:
            raise RuntimeError("no legal actions available")
        best_length = max(length for length, _count, _action in scored)
        best_count = max(count for length, count, _action in scored if length == best_length)
        return [
            action
            for length, count, action in scored
            if length == best_length and count == best_count
        ]

    def _sample_by_center_distance(self, actions: list[int]) -> int:
        center = (self.board_size - 1) / 2.0
        weights = []
        for action in actions:
            row, col = divmod(int(action), self.board_size)
            distance = float(np.hypot(row - center, col - center))
            weights.append(1.0 / (distance + 1.0))
        probs = np.asarray(weights, dtype=np.float64)
        probs /= probs.sum()
        return int(np.random.choice(np.asarray(actions, dtype=np.int64), p=probs))


_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _line_shape(
    board: np.ndarray,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> tuple[int, int]:
    length = 1
    open_ends = 0
    size = board.shape[0]

    for sign in (-1, 1):
        rr = row + sign * dr
        cc = col + sign * dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            length += 1
            rr += sign * dr
            cc += sign * dc
        if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == 0:
            open_ends += 1

    return length, open_ends

