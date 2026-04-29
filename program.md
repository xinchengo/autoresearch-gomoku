# autoresearch-gomoku

This is an autonomous research program for improving a Gomoku agent with PPO self-play.

The project is inspired by `karpathy/autoresearch`, but the loop is different because reinforcement learning runs can last for hours or days and can resume from checkpoints.

## Non-Negotiable Rule

The deployed Gomoku agent must be a pure neural-network policy at inference time.

Allowed at inference:
- one model forward pass
- legal-action masking
- argmax or sampling from the masked policy

Not allowed at inference:
- MCTS
- minimax
- alpha-beta search
- rollouts
- tactical rule search layered on top of the network
- opponent simulation

Training may use PPO self-play and evaluation games, but do not add search-time overhead to the final policy.

## Setup

To set up a new research run:

1. Agree on a run tag, preferably date based, such as `apr29`.
2. Create a branch such as `autoresearch/<tag>`.
3. Read the in-scope files:
   - `README.md` for repository context.
   - `src/gobang/` for the Gymnasium-compatible Gomoku environment.
   - `src/oracle/` for immutable course-agent adapter boundaries.
   - `src/trainer.py` for PPO training.
   - `src/models/` for neural policy-value architectures.
4. Run a tiny smoke test before long experiments:

```bash
uv run python -m src.trainer --tag smoke --board-size 5 --n-in-row 4 --duration 30s --checkpoint-interval 30s --eval-interval 30s --eval-games 2 --device cpu
```

5. Start the first baseline run before modifying research code.

## Mutable And Immutable Files

What you CAN modify:
- `src/trainer.py`
- `src/models/`
- tests that cover changed behavior
- this `program.md` if you are improving the research procedure

What you CANNOT modify during autonomous research:
- `src/gobang/`
- `src/oracle/`
- evaluation semantics in the trainer, except to add adapters for explicitly provided course agents
- the no-search inference rule

The environment and oracle package are the benchmark boundary. Treat them as fixed unless the human explicitly asks for infrastructure changes.

## Experimentation

The goal is to maximize evaluation score against the configured baseline opponents, while preserving simple pure-network inference.

Runs are checkpointed and may last for days. You may choose the run length based on the experiment:

```bash
uv run python -m src.trainer --tag <tag> --duration 6h --checkpoint-interval 30m --eval-interval 30m
```

Resume from a previous checkpoint when continuing a promising line:

```bash
uv run python -m src.trainer --tag <tag> --resume runs/<tag>/checkpoints/latest.pt --duration 6h
```

Each run writes:
- `runs/<tag>/checkpoints/latest.pt`
- milestone checkpoints in `runs/<tag>/checkpoints/`
- `runs/<tag>/metrics.jsonl`
- `runs/<tag>/results.tsv`

Do not commit `runs/`, logs, or `results.tsv`.

## Research Loop

LOOP until interrupted:

1. Inspect current git state and latest metrics.
2. Decide whether to resume a promising checkpoint or start a fresh run.
3. Make one coherent research change.
4. Run a short smoke test if the change touches execution logic.
5. Commit the change before the long run.
6. Launch training with stdout/stderr redirected to a log file.
7. Periodically inspect `metrics.jsonl`, `results.tsv`, and checkpoint health.
8. Keep changes that improve score or materially simplify the system without hurting score.
9. Revert or discard changes that crash, violate pure-network inference, or make results worse without a clear follow-up.

If a long run is interrupted, prefer resuming from `latest.pt` unless the experiment was clearly bad.

## Metrics

The default score is:

```text
score = win_rate + 0.5 * draw_rate
```

The baseline evaluator alternates the neural policy as first and second player against a random legal opponent. If course agents are configured through `src/oracle/course_adapter.py`, they may be added as additional evaluation opponents without changing the no-search rule for the trained policy.

## Simplicity Criterion

All else equal, choose simpler changes.

Good research changes include:
- better PPO targets or advantage normalization
- stronger pure neural architectures
- improved self-play sampling
- curriculum over board sizes
- better checkpoint and resume reliability
- evaluation against stronger provided course agents

Bad research changes include:
- hidden search at inference
- hard-coded tactics in action selection
- brittle reward hacks that only exploit random evaluation
- unbounded complexity for tiny metric gains

