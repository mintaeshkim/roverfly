import numpy as np

from roverfly.math.geometry import hat, rodriguesExpm, vee


def test_hat_and_vee_are_inverses() -> None:
    vector = np.array([1.0, -2.0, 3.0])
    np.testing.assert_array_equal(vee(hat(vector)), vector)


def test_rodrigues_returns_rotation_matrix() -> None:
    rotation = rodriguesExpm(np.array([0.0, 0.0, 1.0]), np.pi / 3)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
