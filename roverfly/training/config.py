"""Training configuration and environment-specific PPO presets."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PpoPreset:
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coefficient: float
    network: tuple[int, ...]
    activation: str
    evaluation_frequency: int
    stop_reward: float
    rollout_steps: int | None = None
    batch_size_per_env: int | None = None
    small_rollout_steps: int = 256
    small_batch_size: int | None = None


PRESETS = {
    "falcon": PpoPreset(1e-4, 0.99, 0.98, 0.05, 0.001, (256, 128, 64), "silu", 2_500, 10_000),
    "mini": PpoPreset(1e-4, 0.99, 0.98, 0.05, 0.001, (256, 128, 64), "silu", 2_500, 10_000),
    "payload": PpoPreset(1e-4, 0.99, 0.98, 0.05, 0.001, (256, 128, 64), "silu", 2_500, 10_000),
    "random": PpoPreset(
        1e-4,
        0.99,
        0.95,
        0.20,
        0.0,
        (256, 256),
        "tanh",
        2_500,
        59_500,
        small_rollout_steps=16_384,
        small_batch_size=8_192,
    ),
}


@dataclass(frozen=True)
class TrainingConfig:
    environment: str = "mini"
    experiment: str = "untitled"
    total_timesteps: int = 100_000_000
    num_envs: int = 8
    device: str = "auto"
    seed: int = 0
    output_dir: Path = Path("runs")
    checkpoint: str | None = None
    visualize: bool = False

    def __post_init__(self) -> None:
        if self.environment not in PRESETS:
            raise ValueError(f"Unknown environment: {self.environment}")
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.visualize and self.num_envs != 1:
            raise ValueError("visualization requires --num-envs 1")

    @property
    def preset(self) -> PpoPreset:
        return PRESETS[self.environment]

    @property
    def run_dir(self) -> Path:
        return self.output_dir.expanduser().resolve() / self.experiment

    @property
    def rollout_steps(self) -> int:
        if self.preset.rollout_steps is not None:
            return self.preset.rollout_steps
        return 64 if self.num_envs >= 16 else self.preset.small_rollout_steps

    @property
    def batch_size(self) -> int:
        if self.preset.batch_size_per_env is not None:
            return self.preset.batch_size_per_env * self.num_envs
        if self.num_envs < 16 and self.preset.small_batch_size is not None:
            return self.preset.small_batch_size
        return min(32 * self.num_envs, 4_096)
