from __future__ import annotations

import argparse

from src.training.config import parse_checkpoint_specs, parse_duration, parse_opponents
from src.training.runner import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO self-play trainer for Gomoku")
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--tag")
    parser.add_argument("--run-dir")
    parser.add_argument("--board-size", type=int)
    parser.add_argument("--n-in-row", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--duration", type=parse_duration)
    parser.add_argument("--checkpoint-interval", type=parse_duration)
    parser.add_argument("--eval-interval", type=parse_duration)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--clip-coef", type=float)
    parser.add_argument("--value-clip-coef", type=float)
    parser.add_argument("--entropy-coef", type=float)
    parser.add_argument("--value-coef", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--eval-games", type=int)
    parser.add_argument("--eval-opponents", type=parse_opponents)
    parser.add_argument("--eval-checkpoints", type=parse_checkpoint_specs)
    parser.add_argument("--resume", default="")
    parser.add_argument("--description")
    parser.add_argument("--curriculum-opponent")
    parser.add_argument("--curriculum-duration", type=parse_duration)
    return parser


def main() -> None:
    train(build_parser().parse_args())
