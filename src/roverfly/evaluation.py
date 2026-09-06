"""Policy evaluation utilities."""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from roverfly.training.runner import make_vector_env


def evaluate(
    model_path: Path,
    *,
    environment: str,
    episodes: int = 10,
    device: str = "auto",
    visualize: bool = False,
) -> tuple[float, float]:
    """Evaluate a saved policy in one environment."""
    model = PPO.load(str(model_path.expanduser()), device=device)
    env = make_vector_env(
        environment,
        num_envs=1,
        seed=0,
        render_mode="human" if visualize else None,
    )
    try:
        mean, std = evaluate_policy(
            model,
            env,
            n_eval_episodes=episodes,
            deterministic=True,
            render=visualize,
        )
        return float(mean), float(std)
    finally:
        env.close()
