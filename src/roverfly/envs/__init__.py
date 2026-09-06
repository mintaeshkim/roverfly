"""Gymnasium environments exposed by RoVerFly."""

from importlib import import_module
from typing import Any

ENVIRONMENTS = {
    "falcon": ("roverfly.envs.falcon", "QuadrotorEnv"),
    "mini": ("roverfly.envs.mini", "QuadrotorMiniEnv"),
    "payload": ("roverfly.envs.payload", "QuadrotorPayloadEnv"),
    "random": ("roverfly.envs.randomized", "QuadrotorRandomEnv"),
}

EXPERIMENTAL_ENVIRONMENTS = {
    "multi": ("roverfly.envs.multi", "QuadrotorMultipleEnv"),
}

_ALL_ENVIRONMENTS = ENVIRONMENTS | EXPERIMENTAL_ENVIRONMENTS


def environment_class(name: str) -> type:
    """Load an environment class by its short CLI name."""
    try:
        module_name, class_name = _ALL_ENVIRONMENTS[name]
    except KeyError as exc:
        choices = ", ".join(_ALL_ENVIRONMENTS)
        raise ValueError(f"Unknown environment {name!r}; choose one of: {choices}") from exc
    return getattr(import_module(module_name), class_name)


def make(name: str, **kwargs: Any):
    """Instantiate an environment by name."""
    return environment_class(name)(**kwargs)


__all__ = ["ENVIRONMENTS", "EXPERIMENTAL_ENVIRONMENTS", "environment_class", "make"]
