from __future__ import annotations

from typing import Protocol

import numpy as np


class OracleAgent(Protocol):
    """Adapter protocol for immutable course-provided Gomoku agents."""

    name: str

    def select_action(self, board: np.ndarray, player: int, legal_mask: np.ndarray) -> int:
        """Return a flat action index for the given board and player."""

