from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


CoursePolicyFn = Callable[[np.ndarray, int, np.ndarray], int]


@dataclass(frozen=True)
class CourseAgentAdapter:
    """Thin wrapper around course agents without importing or modifying them here."""

    name: str
    policy_fn: CoursePolicyFn

    def select_action(self, board: np.ndarray, player: int, legal_mask: np.ndarray) -> int:
        action = int(self.policy_fn(board.copy(), int(player), legal_mask.copy()))
        if action < 0 or action >= legal_mask.size or not bool(legal_mask[action]):
            raise ValueError(f"{self.name} returned illegal action {action}")
        return action


def unavailable_course_agent(name: str = "course-agent") -> CourseAgentAdapter:
    def _missing_policy(_board: np.ndarray, _player: int, _legal_mask: np.ndarray) -> int:
        raise RuntimeError(
            f"{name} is not configured. Wrap your course policy with CourseAgentAdapter."
        )

    return CourseAgentAdapter(name=name, policy_fn=_missing_policy)

