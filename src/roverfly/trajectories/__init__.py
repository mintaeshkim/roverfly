"""Reference trajectories used by the simulation environments."""

from roverfly.trajectories.base import (
    QuinticTrajectory,
    Setpoint,
    SmoothTraj,
    Trajectory,
    random_point_on_sphere,
)
from roverfly.trajectories.basic import CircularTraj, CrazyTrajectory, FullCrazyTrajectory
from roverfly.trajectories.multi import CrazyTrajectoryPayloadMultiple
from roverfly.trajectories.payload import (
    CircularTrajPayload,
    CrazyTrajectoryPayload,
    CrazyTrajectoryPayloadSwing,
    CustomTrajectoryPayloadWindow,
)
from roverfly.trajectories.payload_composite import (
    FullCrazyTrajectoryPayload,
    PredefinedTrajectoryPayload,
)
from roverfly.trajectories.payload_geometric import GeometricTrajectoryPayload, PayloadFigureEight

__all__ = [
    "CircularTraj",
    "CircularTrajPayload",
    "CrazyTrajectory",
    "CrazyTrajectoryPayload",
    "CrazyTrajectoryPayloadMultiple",
    "CrazyTrajectoryPayloadSwing",
    "CustomTrajectoryPayloadWindow",
    "FullCrazyTrajectory",
    "FullCrazyTrajectoryPayload",
    "GeometricTrajectoryPayload",
    "PayloadFigureEight",
    "PredefinedTrajectoryPayload",
    "QuinticTrajectory",
    "Setpoint",
    "SmoothTraj",
    "Trajectory",
    "random_point_on_sphere",
]
