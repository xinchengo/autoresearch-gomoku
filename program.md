# autoresearch-gomoku

Autonomous research program for improving a Gomoku agent with PPO self-play.

This follows the spirit of `karpathy/autoresearch`: the human writes the research protocol here, the agent edits the allowed training code, runs experiments, logs results, and keeps only changes that help. Unlike the upstream 5-minute supervised loop, Gomoku RL runs may last hours or days and should resume from checkpoints.

## Hard Rule

Inference must be pure neural network policy inference:

- allowed: one model forward pass, legal-action masking, argmax or sampling
- forbidden: MCTS, minimax, alpha-beta, rollouts, tactical rule search, opponent simulation, or any other search-time helper

Oracle policies in `src/oracle/` are evaluation opponents only. Do not copy their rules into the trained policy's inference path.

## Setup

For a new run:

1. Pick a run tag such as `apr29` and create `autoresearch/<tag>` from the current main branch.
2. Read `README.md`, `src/gobang/`, `src/oracle/`, `src/training/`, `src/configs/`, `src/algorithms/`, and `src/models/`.
3. Run tests and a smoke run before any long experiment:

```bash
uv run pytest
uv run python -m src.trainer --config smoke
```

4. Establish the current baseline before changing research code.

## Boundaries

You may modify:

- `src/training/`
- `src/configs/`
- `src/algorithms/`
- `src/models/`
- tests for changed behavior
- `program.md` when improving the research protocol

Do not modify during autonomous research unless explicitly instructed:

- `src/gobang/`
- `src/oracle/`
- the evaluation meaning of score/win/draw/loss
- the no-search inference rule

## Experiments

Runs are checkpointed. Choose duration based on the idea:

```bash
uv run python -m src.trainer --tag <tag> --duration 6h --checkpoint-interval 30m --eval-interval 30m
```

Resume promising runs:

```bash
uv run python -m src.trainer --tag <tag> --resume runs/<tag>/checkpoints/latest.pt --duration 6h
```

Evaluate against built-in oracle opponents when no course baseline is available:

```bash
uv run python -m src.trainer --tag <tag> --eval-opponents random,tactical
```

Evaluate against previously promoted neural checkpoints:

```bash
uv run python -m src.trainer --tag <tag> --eval-checkpoints base1=runs/baseline/checkpoints/best.pt,strong-cnn=runs/strong-cnn/checkpoints/best.pt
```

Each run writes `runs/<tag>/checkpoints/`, `metrics.jsonl`, and `results.tsv`. Do not commit `runs/`, logs, or result TSVs.

## Metric

The primary score is averaged across configured opponents:

```text
score = win_rate + 0.5 * draw_rate
```

Higher is better, but no fixed opponent is sufficient. Inspect per-opponent metrics in `metrics.jsonl` and `results.tsv` so a change does not merely exploit a weak oracle or one historical model while regressing against stronger checkpoints.

## Baseline Ladder

Maintain a small ladder of representative neural baselines. Gomoku strength can be highly transitive across many levels, so historical self-play comparisons matter more than any single fixed oracle score.

Promote a checkpoint to the ladder when it is:

- clearly stronger than the current ladder head or fills a useful intermediate strength level
- structurally simple enough to remain a reference point
- stable across both first-player and second-player games
- not merely overfit to `random` or `tactical`

When promoting, keep a named checkpoint path such as `runs/<tag>/checkpoints/best.pt` and include it in future `--eval-checkpoints` runs. Do not add every checkpoint; keep the ladder compact and representative. Prefer 3-8 baselines covering weak, medium, strong, and current-best agents.

`results.tsv` includes aggregate score plus `details_json`. The JSON metrics are the authoritative record for oracle scores and neural-baseline scores such as `ckpt_base1_score` or `ckpt_strong_cnn_win_rate`.

## Logging And Git Loop

LOOP until interrupted:

1. Inspect git state, latest `metrics.jsonl`, latest `results.tsv`, and available checkpoints.
2. Choose one coherent research change.
3. Commit the change before the long run.
4. Run training with output redirected to a log file.
5. If the run crashes, inspect the last log lines; fix simple bugs, otherwise mark the idea as crash/discard.
6. Compare aggregate score, oracle metrics, checkpoint-baseline metrics, stability, and complexity.
7. Keep the commit if it improves the baseline ladder, beats the current head, or simplifies without hurting strength.
8. Promote representative improved checkpoints to the ladder and include them in future evaluations.
9. Revert/discard the commit if it is worse, unstable, too complex for the gain, or violates pure-network inference.

Do not pause after the loop starts. Continue experimenting until the human interrupts. If a long run is interrupted, resume from `latest.pt` unless the idea is clearly bad.

## Research Taste

Prefer simple changes with measurable effect: PPO targets, advantage handling, curricula, network architecture, self-play sampling, checkpoint reliability, or stronger evaluation. Avoid brittle reward hacks, hidden search, hard-coded tactics, and large complexity for tiny gains.
