"""RoVerFly: reinforcement-learning control for quadrotor-payload systems."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("roverfly")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.0.0"

__all__ = ["__version__"]
