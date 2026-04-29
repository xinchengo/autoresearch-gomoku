# Autoresearch for Gomoku

In the *AI3002: Fundamentals of AI & ML* course, I have implemented (with other members of my group) a reinforcement learning agent for the gomoku game.

However, with the advent of `karparthy/autoresearch`, I have decided to implement an autoresearch agent for the same game. I aim to beat the best agent of this course, by using autoresearch.

## File Structure

- `src/`: Contains the source code for the RL framework and the gomoku game.
    - `gobang/`: The Gymnasium-compatible environment for the gomoku game. The N-in-a-row and the board size are configurable.
    - `oracle/`: The baselines, or evaluation methods provided by the repo creator. Should not be modified by the research agent.
    - `trainer.py`
    - `models/`: Model architectures for the gobang agent.
    - ...

## Quick Start

Install dependencies:

```bash
uv sync --extra dev
```

Run the test suite:

```bash
uv run pytest
```

Run a CPU smoke training job:

```bash
uv run python -m src.trainer --tag smoke --board-size 5 --n-in-row 4 --duration 30s --checkpoint-interval 30s --eval-interval 30s --eval-games 2 --device cpu
```

Run a longer checkpointed experiment:

```bash
uv run python -m src.trainer --tag baseline --duration 6h --checkpoint-interval 30m --eval-interval 30m
```

Resume from the latest checkpoint:

```bash
uv run python -m src.trainer --tag baseline --resume runs/baseline/checkpoints/latest.pt --duration 6h
```

The trained policy must remain a pure neural-network policy at inference time: one forward pass, legal-action masking, and argmax or sampling. Do not add MCTS, minimax, rollouts, or other search-time logic.
