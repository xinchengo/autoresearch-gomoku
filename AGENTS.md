# AGENTS.md — Autoresearch Gomoku

PPO self-play RL scaffold for a Gomoku (五子棋) agent. Goal: beat a course-best agent via autonomous research.

> **`program.md`** is the authoritative research protocol. Read it for the full autonomous research loop (git loop, baseline ladder, promotion rules).

## Quick Commands

```bash
uv sync --extra dev          # install deps (Python ≥3.10, torch≥2.1)
uv run pytest                # run all tests (19 tests, ~3s)
uv run python -m src.trainer --config smoke     # 30s CPU smoke run (5×5 board)
uv run python -m src.trainer --tag baseline --duration 6h --checkpoint-interval 30m --eval-interval 30m
```

- Package manager: **uv** (`pyproject.toml` + `uv.lock`). No Poetry/pip.
- Virtual env: `.venv/` (gitignored). `uv sync` manages it.
- No type-checker, linter, formatter, or CI config. No pre-commit hooks.
- No Docker config, no codegen.

## Architecture

```
src/
├── gobang/          # GomokuEnv — Gymnasium env. board_size, n_in_row configurable.
├── oracle/          # Baseline opponents: RandomPolicy, TacticalPolicy, course_adapter.
├── training/        # CLI, runner, config, checkpointing, evaluation.
├── models/          # ActorCriticNet — Conv2d body + policy/value heads. Residual blocks.
├── configs/         # Python config presets (smoke.py, baseline.py). Loaded dynamically.
├── algorithms/      # PPO: collect_self_play (GomokuEnv rollout) + ppo_update (clipped PPO).
├── trainer.py       # Entrypoint: `python -m src.trainer` → CLI → runner.train()
tests/
└── test_*.py        # 19 tests. pytest via `uv run pytest`.
```

### Immutable Modules (during autoresearch)

- `src/gobang/` — Environment. If tests pass here, don't touch.
- `src/oracle/` — Baseline policies (RandomPolicy, TacticalPolicy). **Do not copy their rules into the trained policy.**

### Mutable Modules (allowed to change)

- `src/training/` — CLI, run loop, checkpointing, evaluation, config
- `src/configs/` — Presets: `get_config()` returns `TrainConfig`
- `src/algorithms/` — PPO rollout/update
- `src/models/` — Model architecture
- `tests/` — For changed behavior

## Critical Constraints

**No search-time logic in the trained policy.** The agent must be pure neural network:
- Allowed: one forward pass, legal-action masking, argmax or sampling
- Forbidden: MCTS, minimax, alpha-beta, rollouts, tactical rules, opponent simulation

**Do not commit** `runs/`, `*.log`, `results.tsv` — all gitignored.

## Training & Evaluation

### Run a training session

```bash
uv run python -m src.trainer --tag <tag> --duration 6h --checkpoint-interval 30m --eval-interval 30m
uv run python -m src.trainer --tag <tag> --resume runs/<tag>/checkpoints/latest.pt --duration 6h
```

Output: `runs/<tag>/checkpoints/{latest.pt,best.pt,final.pt,step_N.pt}`, `metrics.jsonl`, `results.tsv`

### Evaluation

```bash
# Against oracle baselines:
uv run python -m src.trainer --tag eval --eval-opponents random,tactical
# Against saved neural checkpoints:
uv run python -m src.trainer --tag eval --eval-checkpoints base=runs/baseline/checkpoints/best.pt,strong=runs/strong/checkpoints/best.pt
```

- Evaluation alternates model as BLACK/WHITE (half games each)
- Score = `win_rate + 0.5 * draw_rate`
- Per-opponent metrics are prefixed (e.g., `random_score`, `ckpt_base_win_rate`)

### Duration format

CLI accepts `30s`, `2m`, `1.5h`, `1d`. Parsed by `parse_duration()` in `src/training/config.py`.

## Technical Details

- **Observation**: `(4, board_size, board_size)` float32 — own stones, opponent stones, legal cells, first-player indicator (all from current player's perspective)
- **Player encoding**: `BLACK=1`, `WHITE=-1`, `EMPTY=0`
- **Illegal moves**: End the game with reward -1 for the offending player
- **Zero-sum returns**: PPO uses perspective returns (`R[t] = r[t] - γ * R[t+1]`), so rewards alternate sign for the opponent
- **Legal-action masking**: `logits.masked_fill(~mask, float_info.min)` before softmax
- **Config presets**: Python modules under `src/configs/`. Must export `get_config() -> TrainConfig` or `CONFIG: TrainConfig | dict`. Loaded via `import_module(f"src.configs.{name}")`.
- **Checkpoints**: `torch.save()` with full dict (`config`, `state`, `model`, `optimizer`, `rng_state`). RNG state (Python, numpy, torch, CUDA) is saved for exact resume.
- **`latest.pt`**: Always overwritten with the most recent checkpoint.
- **`best.pt`**: Written when `score > state.best_score` during evaluation.
- **`final.pt`**: Written at training end (even 0s duration).
- **No CI**: No GitHub Actions/workflows. All runs are local/long-running.

### Baseline Ladder

Maintain 3-8 representative neural baselines covering weak→strong. Gomoku strength is transitive, so historical self-play comparisons matter more than any single oracle score. Promote a checkpoint when it is:
- Clearly stronger than the current ladder head (or fills a useful intermediate strength level)
- Structurally simple enough
- Stable across both BLACK and WHITE games
- Not overfit to `random` or `tactical`

Promoted checkpoints go into `--eval-checkpoints` (e.g., `l1=runs/v1/checkpoints/best.pt`). `results.tsv` JSON metrics are the authoritative record.

### Default hyperparameters (12×12, n_in_row=5)

| Param | Value |
|---|---|
| rollout_steps | 1024 |
| ppo_epochs | 4 |
| minibatch_size | 256 |
| gamma | 0.99 |
| clip_coef | 0.2 |
| entropy_coef | 0.02 |
| value_coef | 0.5 |
| learning_rate | 3e-4 |
| channels | 64 |
| blocks | 4 |

## Code Conventions

- `from __future__ import annotations` at top of every file
- All type annotations use full types (no implicit `Any`)
- Functions have explicit `-> None` return type
- Dataclasses for config/state: `TrainConfig`, `TrainState`, `GomokuConfig`
- Protocol class `OracleAgent` for opponent interface
- PPO algorithm functions are standalone (not class methods): `collect_self_play()`, `ppo_update()`
- Tests use `assert`, parametrize, and temporary directories (`tmp_path` fixture)
