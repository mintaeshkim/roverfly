import numpy as np

from roverfly import trajectories


def test_setpoint_is_stationary() -> None:
    point = np.array([1.0, 2.0, 3.0])
    position, velocity, acceleration = trajectories.Setpoint(point).get(2.0)
    np.testing.assert_array_equal(position, point)
    np.testing.assert_array_equal(velocity, np.zeros(3))
    np.testing.assert_array_equal(acceleration, np.zeros(3))


def test_single_vehicle_trajectory_contract() -> None:
    values = trajectories.CrazyTrajectory(tf=1).get(0.5)
    assert len(values) == 3
    assert all(value.shape == (3,) for value in values)


def test_payload_trajectory_contract() -> None:
    values = trajectories.CrazyTrajectoryPayload(tf=1).get(0.5)
    assert len(values) == 7
    assert all(value.shape == (3,) for value in values)


def test_multi_payload_trajectory_contract() -> None:
    values = trajectories.CrazyTrajectoryPayloadMultiple(tf=1).get(0.5)
    assert len(values) == 9
    assert all(value.shape == (3,) for value in values)
