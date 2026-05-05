from src.training.config import TrainConfig


def get_config() -> TrainConfig:
    return TrainConfig(
        tag="baseline",
        board_size=12,
        n_in_row=5,
        duration_seconds=3600.0,
        checkpoint_interval_seconds=1800.0,
        eval_interval_seconds=1800.0,
        rollout_steps=4096,
        ppo_epochs=8,
        minibatch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        channels=64,
        blocks=4,
        eval_games=40,
        eval_opponents=("random", "tactical"),
        eval_checkpoints=(),
        description="baseline PPO self-play",
    )
