import math

import numpy as np


def hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def rodriguesExpm(v, θ):
    if np.linalg.norm(v) <= 1e-4:
        v = np.array([0, 1, 0])
    K = hat(v / np.linalg.norm(v))
    if abs(θ) <= 1e-4:
        return np.eye(3)
    else:
        return np.eye(3) + np.sin(θ) * K + (1 - np.cos(θ)) * K @ K


def expmTaylorExpansion(M, order=2):
    R = np.eye(3)
    for i in range(1, order + 1):
        R += np.linalg.matrix_power(M, i) / math.factorial(i)
