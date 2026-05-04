from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _Env:
        metadata: dict[str, list[str]] = {}

        def reset(self, *, seed: int | None = None) -> None:
            if seed is not None:
                np.random.seed(seed)

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: type) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


EMPTY = 0
BLACK = 1
WHITE = -1


def check_winner(board: np.ndarray, n_in_row: int, last_move: int | None = None) -> int:
    """Return BLACK, WHITE, or EMPTY if there is no winner."""
    size = int(board.shape[0])
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))

    if last_move is None:
        candidates = np.argwhere(board != EMPTY)
    else:
        row, col = divmod(int(last_move), size)
        candidates = np.array([[row, col]], dtype=np.int64)

    for row, col in candidates:
        player = int(board[row, col])
        if player == EMPTY:
            continue
        for dr, dc in directions:
            count = 1
            for sign in (-1, 1):
                rr = int(row) + sign * dr
                cc = int(col) + sign * dc
                while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= n_in_row:
                return player
    return EMPTY


@dataclass(frozen=True)
class GomokuConfig:
    board_size: int = 12
    n_in_row: int = 5
    illegal_move_ends_game: bool = True


class GomokuEnv(gym.Env):
    """Gymnasium-compatible Gomoku/Gobang environment.

    The observation is always from the current player's perspective:
    own stones, opponent stones, legal cells, and a first-player indicator plane.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        board_size: int = 12,
        n_in_row: int = 5,
        illegal_move_ends_game: bool = True,
    ) -> None:
        super().__init__()
        if board_size < 3:
            raise ValueError("board_size must be at least 3")
        if n_in_row < 3 or n_in_row > board_size:
            raise ValueError("n_in_row must be between 3 and board_size")

        self.config = GomokuConfig(board_size, n_in_row, illegal_move_ends_game)
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, board_size, board_size),
            dtype=np.float32,
        )
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = BLACK
        self.move_count = 0
        self.winner = EMPTY
        self.last_move: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.move_count = 0
        self.winner = EMPTY
        self.last_move = None
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        terminated = False
        reward = 0.0

        if action < 0 or action >= self.action_space.n:
            return self._illegal_result(action)

        row, col = divmod(action, self.board_size)
        if self.board[row, col] != EMPTY:
            return self._illegal_result(action)

        acting_player = self.current_player
        self.board[row, col] = acting_player
        self.move_count += 1
        self.last_move = action

        winner = check_winner(self.board, self.n_in_row, action)
        if winner != EMPTY:
            self.winner = winner
            terminated = True
            reward = 1.0
        elif self.move_count == self.action_space.n:
            terminated = True
            reward = 0.0
        else:
            self.current_player = -self.current_player

        return self._observation(), reward, terminated, False, self._info()

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.action_mask())

    def action_mask(self) -> np.ndarray:
        return (self.board.reshape(-1) == EMPTY)

    def render(self) -> str:
        symbols = {BLACK: "X", WHITE: "O", EMPTY: "."}
        rows = [" ".join(symbols[int(v)] for v in row) for row in self.board]
        return "\n".join(rows)

    def _observation(self) -> np.ndarray:
        own = self.board == self.current_player
        opponent = self.board == -self.current_player
        legal = self.board == EMPTY
        first_player = np.full_like(self.board, self.current_player == BLACK, dtype=np.float32)
        return np.stack(
            [
                own.astype(np.float32),
                opponent.astype(np.float32),
                legal.astype(np.float32),
                first_player,
            ],
            axis=0,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.action_mask().copy(),
            "current_player": int(self.current_player),
            "winner": int(self.winner),
            "last_move": self.last_move,
            "move_count": int(self.move_count),
        }

    def _illegal_result(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.config.illegal_move_ends_game:
            info = self._info()
            info["illegal_action"] = action
            return self._observation(), -1.0, False, False, info

        self.winner = -self.current_player
        info = self._info()
        info["illegal_action"] = action
        return self._observation(), -1.0, True, False, info
