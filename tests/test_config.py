from pathlib import Path

import pytest

from roverfly.training.config import TrainingConfig


def test_default_rollout_and_batch_sizes() -> None:
    config = TrainingConfig()
    assert config.rollout_steps == 256
    assert config.batch_size == 256


def test_environment_specific_training_sizes() -> None:
    random_config = TrainingConfig(environment="random")
    parallel_config = TrainingConfig(num_envs=16)

    assert (random_config.rollout_steps, random_config.batch_size) == (16_384, 8_192)
    assert (parallel_config.rollout_steps, parallel_config.batch_size) == (64, 512)


def test_experimental_environment_cannot_be_trained() -> None:
    with pytest.raises(ValueError, match="Unknown environment"):
        TrainingConfig(environment="multi")


def test_run_directory_is_scoped_to_experiment(tmp_path: Path) -> None:
    config = TrainingConfig(experiment="paper", output_dir=tmp_path)
    assert config.run_dir == tmp_path / "paper"


@pytest.mark.parametrize("field", ["num_envs", "total_timesteps"])
def test_positive_training_counts(field: str) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{field: 0})


def test_visualization_requires_one_environment() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(visualize=True, num_envs=2)
