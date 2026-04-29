from src.training.config import TrainConfig


def get_config() -> TrainConfig:
    return TrainConfig(
        tag="smoke",
        board_size=5,
        n_in_row=4,
        duration_seconds=30.0,
        checkpoint_interval_seconds=30.0,
        eval_interval_seconds=30.0,
        rollout_steps=64,
        ppo_epochs=1,
        minibatch_size=32,
        channels=16,
        blocks=1,
        eval_games=2,
        eval_opponents=("random",),
        eval_checkpoints=(),
        device="cpu",
        description="small-board PPO smoke run",
    )
