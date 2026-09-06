import pytest

from roverfly.envs import ENVIRONMENTS, EXPERIMENTAL_ENVIRONMENTS, make


@pytest.mark.integration
@pytest.mark.parametrize("environment", ENVIRONMENTS | EXPERIMENTAL_ENVIRONMENTS)
def test_environment_reset_and_step(environment: str) -> None:
    env = make(environment, render_mode=None)
    try:
        observation, _ = env.reset(seed=0)
        next_observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        assert env.observation_space.contains(next_observation)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()
