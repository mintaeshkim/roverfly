"""Reference trajectories for cooperative multi-quadrotor payloads."""

import numpy as np

from roverfly.math.geometry import rodriguesExpm
from roverfly.trajectories.base import Trajectory


class CrazyTrajectoryPayloadMultiple(Trajectory):
    def __init__(self, tf=10, ax=5, ay=5, az=5, f1=0.5, f2=0.5, f3=0.5):
        super().__init__(tf)
        alpha_param, beta_param = 5.0, 5.0
        self.ax = ax * np.random.beta(alpha_param, beta_param)
        self.ay = ay * np.random.beta(alpha_param, beta_param)
        self.az = az * np.random.beta(alpha_param, beta_param)
        self.f1 = f1 * np.random.beta(alpha_param, beta_param)
        self.f2 = f2 * np.random.beta(alpha_param, beta_param)
        self.f3 = f3 * np.random.beta(alpha_param, beta_param)
        self.phix = np.random.choice([np.pi / 2, 3 * np.pi / 2])
        self.phiy = np.random.choice([np.pi / 2, 3 * np.pi / 2])
        self.phiz = np.random.choice([np.pi / 2, 3 * np.pi / 2])

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0, 0, 1])

    def get(self, t):
        w1 = 2 * np.pi * self.f1
        w2 = 2 * np.pi * self.f2
        w3 = 2 * np.pi * self.f3

        x = np.array(
            [
                self.ax * (1 - np.cos(w1 * t + self.phix)),
                self.ay * (1 - np.cos(w2 * t + self.phiy)),
                self.az * (1 - np.cos(w3 * t + self.phiz)),
            ]
        )
        v = np.array(
            [
                self.ax * np.sin(w1 * t + self.phix) * w1,
                self.ay * np.sin(w2 * t + self.phiy) * w2,
                self.az * np.sin(w3 * t + self.phiz) * w3,
            ]
        )
        a = np.array(
            [
                self.ax * np.cos(w1 * t + self.phix) * w1 * w1,
                self.ay * np.cos(w2 * t + self.phiy) * w2 * w2,
                self.az * np.cos(w3 * t + self.phiz) * w3 * w3,
            ]
        )
        da = np.array(
            [
                -self.ax * np.sin(w1 * t + self.phix) * w1 * w1 * w1,
                -self.ay * np.sin(w2 * t + self.phiy) * w2 * w2 * w2,
                -self.az * np.sin(w3 * t + self.phiz) * w3 * w3 * w3,
            ]
        )
        d2a = np.array(
            [
                -self.ax * np.cos(w1 * t + self.phix) * w1 * w1 * w1 * w1,
                -self.ay * np.cos(w2 * t + self.phiy) * w2 * w2 * w2 * w2,
                -self.az * np.cos(w3 * t + self.phiz) * w3 * w3 * w3 * w3,
            ]
        )

        Tp = -self.mP * (a + self.g * self.e3)
        norm_Tp = np.linalg.norm(Tp)
        q = Tp / norm_Tp

        dTp = -self.mP * da
        dnorm_Tp = 1 / norm_Tp * np.dot(Tp, dTp)
        dq = (dTp - q * dnorm_Tp) / norm_Tp

        d2Tp = -self.mP * d2a
        d2norm_Tp = (np.dot(dTp, dTp) + np.dot(Tp, d2Tp) - dnorm_Tp**2) / norm_Tp
        d2q = (d2Tp - dq * dnorm_Tp - q * d2norm_Tp - dq * dnorm_Tp) / norm_Tp

        q0 = rodriguesExpm(np.array([0, 1, 0]), np.pi / 6) @ q
        q1 = rodriguesExpm(np.array([0, 1, 0]), -np.pi / 6) @ q

        return x, v, a, da, q0, q1, q, dq, d2q

    def plot3d_payload_multiple(self):
        import matplotlib.pyplot as plt

        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        q0 = np.empty((0, 3))
        q1 = np.empty((0, 3))
        q = np.empty((0, 3))
        for t in T:
            x_, _, _, _, q0_, q1_, q_, _, _ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)
            q0 = np.append(q0, np.array([q0_]), axis=0)
            q1 = np.append(q1, np.array([q1_]), axis=0)
            q = np.append(q, np.array([q_]), axis=0)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label="Trajectory", color="b", linewidth=2)
        ax.set_title("3D Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        max_range = (
            np.array(
                [
                    x[:, 0].max() - x[:, 0].min(),
                    x[:, 1].max() - x[:, 1].min(),
                    x[:, 2].max() - x[:, 2].min(),
                ]
            ).max()
            / 2.0
        )
        mid_x = (x[:, 0].max() + x[:, 0].min()) * 0.5
        mid_y = (x[:, 1].max() + x[:, 1].min()) * 0.5
        mid_z = (x[:, 2].max() + x[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        step = len(T) // 20
        for i in range(0, len(T), step):
            ax.quiver(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                -q0[i, 0],
                -q0[i, 1],
                -q0[i, 2],
                length=0.5,
                normalize=True,
                color="g",
                arrow_length_ratio=0.01,
            )
            ax.quiver(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                -q1[i, 0],
                -q1[i, 1],
                -q1[i, 2],
                length=0.5,
                normalize=True,
                color="b",
                arrow_length_ratio=0.01,
            )
            ax.quiver(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                -q[i, 0],
                -q[i, 1],
                -q[i, 2],
                length=0.5,
                normalize=True,
                color="r",
                arrow_length_ratio=0.01,
            )

        plt.show()
