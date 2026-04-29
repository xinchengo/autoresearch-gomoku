from __future__ import annotations

from src.training.cli import build_parser, main
from src.training.runner import train

__all__ = ["build_parser", "main", "train"]


if __name__ == "__main__":
    main()
