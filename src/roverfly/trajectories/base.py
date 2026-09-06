"""Base classes and interpolation trajectories."""

import warnings

import numpy as np


def random_point_on_sphere(radius):
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(2 * np.random.uniform() - 1)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


class Trajectory:
    def __init__(self, tf=10):
        self._tf = tf

    def compute_traj_params(self):
        raise NotImplementedError

    def get(self, t):
        raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt

        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        v = np.empty((0, 3))
        a = np.empty((0, 3))
        for t in T:
            if len(self.get(t)) == 7:
                x_, v_, a_, _, _, _, _ = self.get(t)  # Payload
            if len(self.get(t)) == 3:
                x_, v_, a_ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)
            v = np.append(v, np.array([v_]), axis=0)
            a = np.append(a, np.array([a_]), axis=0)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs[0].plot(T, x[:, 0], "r", linewidth=2, label="x")
        axs[0].plot(T, x[:, 1], "g", linewidth=2, label="y")
        axs[0].plot(T, x[:, 2], "b", linewidth=2, label="z")
        axs[0].set_title("Position")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(T, v[:, 0], ":r", linewidth=2, label="vx")
        axs[1].plot(T, v[:, 1], ":g", linewidth=2, label="vy")
        axs[1].plot(T, v[:, 2], ":b", linewidth=2, label="vz")
        axs[1].set_title("Velocity")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(T, a[:, 0], "--r", linewidth=2, label="ax")
        axs[2].plot(T, a[:, 1], "--g", linewidth=2, label="ay")
        axs[2].plot(T, a[:, 2], "--b", linewidth=2, label="az")
        axs[2].set_title("Acceleration")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    def plot3d(self):
        import matplotlib.pyplot as plt

        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        for t in T:
            x_, _, _ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label="Trajectory", color="b", linewidth=2)
        ax.set_title("3D Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    def plot3d_payload(self, save_path=None):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # Font and style settings
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["axes.labelsize"] = 14
        mpl.rcParams["xtick.labelsize"] = 12
        mpl.rcParams["ytick.labelsize"] = 12
        mpl.rcParams["legend.fontsize"] = 12
        plt.style.use("seaborn-v0_8-whitegrid")

        # Sample points
        T = np.linspace(0, self._tf, 100)
        x = np.empty((0, 3))
        q = np.empty((0, 3))
        for t in T:
            x_, _, _, _, q_, _, _ = self.get(t)
            x = np.append(x, [x_], axis=0)
            q = np.append(q, [q_], axis=0)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectory
        ax.plot(x[:, 0], x[:, 1], x[:, 2], color="#0055a4", linewidth=2.0)

        # Draw arrows for cable direction (from payload to quadrotor)
        step = len(T) // 25
        for i in range(0, len(T), step):
            ax.quiver(
                x[i, 0],
                x[i, 1],
                x[i, 2],
                -q[i, 0],
                -q[i, 1],
                -q[i, 2],
                length=0.4,
                color="crimson",
                linewidth=1.0,
                arrow_length_ratio=0.1,
                normalize=True,
            )

        # Start and End points
        ax.scatter(*x[0], color="green", s=25, label="Start / End")

        # Set labels
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_zlabel(r"$z$ [m]")

        # Equal aspect ratio
        max_range = np.array([x[:, 0].ptp(), x[:, 1].ptp(), x[:, 2].ptp()]).max() / 2.0
        mid_x = (x[:, 0].max() + x[:, 0].min()) * 0.5
        mid_y = (x[:, 1].max() + x[:, 1].min()) * 0.5
        mid_z = (x[:, 2].max() + x[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Camera angle for better view
        ax.view_init(elev=25, azim=-45)

        # Hide grid planes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Legend (optional)
        ax.legend(loc="upper left")

        # Tight layout and optional save
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.show()

    def plot3d_payload_geometric(self):
        import matplotlib.pyplot as plt

        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        q = np.empty((0, 3))
        for t in T:
            x_, _, _, _, _, _, _, q_, _, _, _, _ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)
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
                -q[i, 0],
                -q[i, 1],
                -q[i, 2],
                length=0.5,
                normalize=True,
                color="r",
                arrow_length_ratio=0.01,
            )

        plt.show()


class Setpoint(Trajectory):
    def __init__(self, setpoint, tf=10):
        super().__init__(tf)
        self._xf = setpoint

    def get(self, t):
        return self._xf, np.zeros(3), np.zeros(3)


class SmoothTraj(Trajectory):
    def __init__(self, x0, xf, tf):
        super().__init__(tf)
        self._x0 = x0
        self._xf = xf
        self._pos_params = []
        self._vel_params = []
        self._acc_params = []

        self._t = lambda phase: np.array([1.0, phase, phase**2, phase**3, phase**4, phase**5])

        self.compute_traj_params()

    def compute_traj_params(self):
        raise NotImplementedError

    def get(self, t):
        if t >= self._tf:
            return self._xf, np.zeros(3), np.zeros(3)
        elif t < 0:
            warnings.warn("Time cannot be negative")
            return self._x0, np.zeros(3), np.zeros(3)
        else:
            phase = t / self._tf
            return (
                (np.array([self._t(phase)]) @ self._pos_params)[0],
                (np.array([self._t(phase)]) @ self._vel_params)[0],
                (np.array([self._t(phase)]) @ self._acc_params)[0],
            )


class QuinticTrajectory(Trajectory):
    def __init__(self, tf, x0, xf):
        super().__init__(tf)
        self.tf = tf
        self.x0 = np.array(x0)
        self.xf = np.array(xf)

    def compute(self, t):
        tau = np.clip(t / self.tf, 0, 1)

        s = 6 * tau**5 - 15 * tau**4 + 10 * tau**3
        ds = (30 * tau**4 - 60 * tau**3 + 30 * tau**2) / self.tf
        dds = (120 * tau**3 - 180 * tau**2 + 60 * tau) / self.tf**2

        position = (1 - s) * self.x0 + s * self.xf
        velocity = ds * (self.xf - self.x0)
        acceleration = dds * (self.xf - self.x0)

        return position, velocity, acceleration

    def get(self, t):
        return self.compute(t)
