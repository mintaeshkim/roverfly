"""Single training pipeline for every RoVerFly environment."""

from collections.abc import Callable
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from roverfly.envs import environment_class
from roverfly.training.callbacks import SubrewardCallback
from roverfly.training.config import TrainingConfig


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linearly decay a scalar from its initial value to zero."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def _make_env(environment: str, *, seed: int, render_mode: str | None):
    env_type = environment_class(environment)

    def initialize():
        env = env_type(render_mode=render_mode, env_num=seed)
        env.reset(seed=seed)
        return env

    return initialize


def make_vector_env(
    environment: str,
    *,
    num_envs: int,
    seed: int,
    render_mode: str | None = None,
):
    set_random_seed(seed)
    factories = [
        _make_env(environment, seed=seed + index, render_mode=render_mode)
        for index in range(num_envs)
    ]
    return VecMonitor(DummyVecEnv(factories))


def resolve_checkpoint(checkpoint: str, output_dir: Path) -> Path:
    """Resolve an explicit path or another run's best/final model."""
    requested = Path(checkpoint).expanduser()
    candidates = [
        requested,
        output_dir / checkpoint / "checkpoints" / "best_model.zip",
        output_dir / checkpoint / "final_model.zip",
        Path("train") / "saved_models" / f"saved_model_{checkpoint}" / "best_model.zip",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.with_suffix(".zip").is_file():
            return candidate.with_suffix(".zip").resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def train(config: TrainingConfig) -> Path:
    """Train a PPO policy and return the final model path."""
    config.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    render_mode = "human" if config.visualize else None
    env = None
    eval_env = None
    try:
        env = make_vector_env(
            config.environment,
            num_envs=config.num_envs,
            seed=config.seed,
            render_mode=render_mode,
        )
        eval_env = make_vector_env(
            config.environment,
            num_envs=1,
            seed=config.seed + 10_000,
            render_mode=None,
        )

        activation = {"silu": torch.nn.SiLU, "tanh": torch.nn.Tanh}[config.preset.activation]
        policy_kwargs = {
            "activation_fn": activation,
            "net_arch": {
                "pi": list(config.preset.network),
                "vf": list(config.preset.network),
            },
        }
        model = PPO(
            "MlpPolicy",
            env=env,
            learning_rate=config.preset.learning_rate,
            n_steps=config.rollout_steps,
            batch_size=config.batch_size,
            gamma=config.preset.gamma,
            gae_lambda=config.preset.gae_lambda,
            clip_range=linear_schedule(config.preset.clip_range),
            ent_coef=config.preset.entropy_coefficient,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(config.run_dir / "tensorboard"),
            device=config.device,
            seed=config.seed,
            verbose=1,
        )

        if config.checkpoint:
            model.set_parameters(str(resolve_checkpoint(config.checkpoint, config.output_dir)))

        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=config.preset.stop_reward,
            verbose=1,
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=stop_callback,
            eval_freq=max(config.preset.evaluation_frequency // config.num_envs, 1),
            best_model_save_path=str(checkpoint_dir),
            log_path=str(config.run_dir / "evaluation"),
            deterministic=True,
            verbose=1,
        )

        model.learn(
            total_timesteps=config.total_timesteps,
            progress_bar=True,
            callback=CallbackList([SubrewardCallback(), eval_callback]),
        )
        final_model = config.run_dir / "final_model"
        model.save(final_model)
        return final_model.with_suffix(".zip")
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()
