from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RandomPolicy:
    name: str = "random"

    def select_action(self, board: np.ndarray, player: int, legal_mask: np.ndarray) -> int:
        del board, player
        legal = np.flatnonzero(legal_mask)
        if legal.size == 0:
            raise RuntimeError("no legal actions available")
        return int(np.random.choice(legal))

