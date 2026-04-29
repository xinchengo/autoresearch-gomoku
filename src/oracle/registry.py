from __future__ import annotations

from src.oracle.base import OracleAgent
from src.oracle.random_policy import RandomPolicy
from src.oracle.tactical_policy import TacticalPolicy


def make_oracle(name: str, board_size: int, n_in_row: int) -> OracleAgent:
    if name == "random":
        return RandomPolicy()
    if name == "tactical":
        return TacticalPolicy(board_size=board_size, n_in_row=n_in_row)
    raise ValueError(f"unknown oracle policy: {name}")

