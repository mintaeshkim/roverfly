"""Composite and predefined payload trajectories."""

import numpy as np

from roverfly.trajectories.base import QuinticTrajectory, Trajectory


class FullCrazyTrajectoryPayload(Trajectory):
    def __init__(self, traj, tf=40):
        """
        Full trajectory that includes:
        1. Smooth takeoff (0-5s) using a QuinticTrajectory.
        2. Crazy trajectory with payload (5-35s).
        3. Smooth landing (35-40s) using a QuinticTrajectory.

        Args:
            traj (Trajectory): An instance of CrazyTrajectoryPayload.
            tf (float): Total duration of the trajectory (default: 45s).
        """
        super().__init__(tf)
        self.crazy_traj = traj
        self.takeoff_height = 3.0  # Target height for takeoff and landing

        # Define takeoff and landing trajectories using QuinticTrajectory
        self.takeoff_traj = QuinticTrajectory(tf=5, x0=[0, 0, 0], xf=[0, 0, self.takeoff_height])
        self.landing_traj = QuinticTrajectory(tf=5, x0=[0, 0, self.takeoff_height], xf=[0, 0, 0])

    def get(self, t):
        """Returns position, velocity, acceleration, jerk, payload orientation, and its derivatives."""
        if t < 5:
            position, velocity, acceleration = self.takeoff_traj.get(t)
            jerk = np.zeros(3)
            q = np.array([0, 0, -1])  # Assume vertical alignment during takeoff
            dq = np.zeros(3)
            d2q = np.zeros(3)
            return position, velocity, acceleration, jerk, q, dq, d2q
        elif t < self._tf - 5:
            crazy_x, crazy_v, crazy_a, crazy_da, crazy_q, crazy_dq, crazy_d2q = self.crazy_traj.get(
                t - 5
            )
            crazy_x[2] += self.takeoff_height  # Shift trajectory to hover at 1.5m height
            return crazy_x, crazy_v, crazy_a, crazy_da, crazy_q, crazy_dq, crazy_d2q
        else:
            position, velocity, acceleration = self.landing_traj.get(t - (self._tf - 5))
            jerk = np.zeros(3)
            q = np.array([0, 0, -1])  # Assume vertical alignment during landing
            dq = np.zeros(3)
            d2q = np.zeros(3)
            return position, velocity, acceleration, jerk, q, dq, d2q


class PredefinedTrajectoryPayload(Trajectory):
    def __init__(self, tf=30, type="hover"):
        super().__init__(tf)
        self.type = type
        sign = np.random.choice([-1, 1], size=3)
        if "swing" in self.type:
            sign[-1] = -1

        trajs = {
            "crazy_1": {
                "ax": 1.0,
                "ay": 1.5,
                "az": 0.5,
                "f1": 0.2,
                "f2": 0.15,
                "f3": 0.1,
                "phix": 0,
                "phiy": 0,
                "phiz": 0,
            },
            "crazy_2": {
                "ax": 1.5,
                "ay": 1.0,
                "az": 0.5,
                "f1": 0.15,
                "f2": 0.2,
                "f3": 0.1,
                "phix": 0,
                "phiy": 0,
                "phiz": 0,
            },
            "crazy_3": {
                "ax": 0.5,
                "ay": 1.5,
                "az": 1.0,
                "f1": 0.3,
                "f2": 0.2,
                "f3": 0.1,
                "phix": 0,
                "phiy": 0,
                "phiz": 0,
            },
            "crazy_4": {
                "ax": 1.5,
                "ay": 0.5,
                "az": 1.0,
                "f1": 0.2,
                "f2": 0.3,
                "f3": 0.1,
                "phix": 0,
                "phiy": 0,
                "phiz": np.pi / 2,
            },
            "swing_1": {
                "ax": 1.0,
                "ay": 0.0,
                "az": 0.05,
                "f1": 0.3,
                "f2": 0.0,
                "f3": 0.6,
                "phix": 0,
                "phiy": 0,
                "phiz": np.pi / 2,
            },
            "swing_2": {
                "ax": 0.0,
                "ay": 1.0,
                "az": 0.05,
                "f1": 0.0,
                "f2": 0.3,
                "f3": 0.6,
                "phix": 0,
                "phiy": 0,
                "phiz": np.pi / 2,
            },
            "swing_3": {
                "ax": 0.5,
                "ay": 0.0,
                "az": 0.05,
                "f1": 0.5,
                "f2": 0.0,
                "f3": 1.0,
                "phix": 0,
                "phiy": 0,
                "phiz": np.pi / 2,
            },
            "swing_4": {
                "ax": 0.0,
                "ay": 0.5,
                "az": 0.05,
                "f1": 0.5,
                "f2": 0.5,
                "f3": 1.0,
                "phix": 0,
                "phiy": 0,
                "phiz": 0,
            },
            "circle_1": {
                "ax": 0.25,
                "ay": 0.25,
                "az": 0.0,
                "f1": 1.0,
                "f2": 1.0,
                "f3": 0.0,
                "phix": 0,
                "phiy": np.pi / 2,
                "phiz": 0,
            },
            "circle_2": {
                "ax": 0.5,
                "ay": 0.5,
                "az": 0.0,
                "f1": 0.5,
                "f2": 0.5,
                "f3": 0.0,
                "phix": 0,
                "phiy": np.pi / 2,
                "phiz": 0,
            },
            "circle_3": {
                "ax": 0.75,
                "ay": 0.75,
                "az": 0.0,
                "f1": 0.3,
                "f2": 0.3,
                "f3": 0.0,
                "phix": 0,
                "phiy": np.pi / 2,
                "phiz": 0,
            },
            "circle_4": {
                "ax": 1.0,
                "ay": 1.0,
                "az": 0.0,
                "f1": 0.25,
                "f2": 0.25,
                "f3": 0.0,
                "phix": 0,
                "phiy": np.pi / 2,
                "phiz": 0,
            },
            "hover": {
                "ax": 0.0,
                "ay": 0.0,
                "az": 0.0,
                "f1": 0.0,
                "f2": 0.0,
                "f3": 0.0,
                "phix": 0,
                "phiy": 0,
                "phiz": 0,
            },
        }

        traj = trajs[self.type]

        self.ax = sign[0] * traj["ax"]
        self.ay = sign[1] * traj["ay"]
        self.az = sign[2] * traj["az"]

        self.f1, self.f2, self.f3 = traj["f1"], traj["f2"], traj["f3"]
        self.phix, self.phiy, self.phiz = traj["phix"], traj["phiy"], traj["phiz"]

        self.v_max = 4.0
        self.w1, self.w2, self.w3 = [2 * np.pi * f for f in (self.f1, self.f2, self.f3)]

        if max(self.ax * self.w1, self.ay * self.w2, self.az * self.w3) > self.v_max:
            scaling_factor = (
                max(self.ax * self.w1, self.ay * self.w2, self.az * self.w3) / self.v_max
            )
            self.ax *= scaling_factor
            self.ay *= scaling_factor
            self.az *= scaling_factor

        self.mP = 0.1
        self.g = 9.81
        self.e3 = np.array([0, 0, 1])

    def __str__(self):
        return (
            f"Predefined Trajectory {self.type}:\n"
            f"  ax = {self.ax:.3f}, ay = {self.ay:.3f}, az = {self.az:.3f}\n"
            f"  f1 = {self.f1:.3f}, f2 = {self.f2:.3f}, f3 = {self.f3:.3f}\n"
            f"  phix = {self.phix:.3f}, phiy = {self.phiy:.3f}, phiz = {self.phiz:.3f}"
        )

    def window(self, t):
        """Window function for smooth velocity transitions at t=3s and t=tf-3s"""
        t_start, t_end = 5, self._tf - 5
        transition_duration = 3.0  # Extended transition for smoothness
        if t < t_start or t > t_end:
            return 0  # Hovering state (no movement)
        elif t_start <= t < t_start + transition_duration:
            x = (t - t_start) / transition_duration
            return 3 * x**2 - 2 * x**3  # Smoothstep function
        elif t_end - transition_duration < t <= t_end:
            x = (t_end - t) / transition_duration
            return 3 * x**2 - 2 * x**3  # Smoothstep function (reverse)
        return 1  # Full trajectory motion

    def d_window(self, t):
        """Derivative of the window function for velocity adjustment"""
        t_start, t_end = 5, self._tf - 5
        transition_duration = 3.0
        if t_start <= t < t_start + transition_duration:
            x = (t - t_start) / transition_duration
            return (6 * x - 6 * x**2) / transition_duration
        elif t_end - transition_duration < t <= t_end:
            x = (t_end - t) / transition_duration
            return (-6 * x + 6 * x**2) / transition_duration
        return 0  # No velocity change in hovering

    def compute(self, t):
        """Compute trajectory with smooth transition (now using sin instead of 1 - cos)"""
        w1, w2, w3 = [2 * np.pi * f for f in (self.f1, self.f2, self.f3)]
        win = self.window(t)
        d_win = self.d_window(t)

        x = np.array(
            [
                win * self.ax * np.sin(w1 * t + self.phix),
                win * self.ay * np.sin(w2 * t + self.phiy),
                win * self.az * np.sin(w3 * t + self.phiz),
            ]
        )
        v = np.array(
            [
                win * self.ax * np.cos(w1 * t + self.phix) * w1
                + d_win * self.ax * np.sin(w1 * t + self.phix),
                win * self.ay * np.cos(w2 * t + self.phiy) * w2
                + d_win * self.ay * np.sin(w2 * t + self.phiy),
                win * self.az * np.cos(w3 * t + self.phiz) * w3
                + d_win * self.az * np.sin(w3 * t + self.phiz),
            ]
        )
        a = np.array(
            [
                -win * self.ax * np.sin(w1 * t + self.phix) * w1**2
                + 2 * d_win * self.ax * np.cos(w1 * t + self.phix) * w1,
                -win * self.ay * np.sin(w2 * t + self.phiy) * w2**2
                + 2 * d_win * self.ay * np.cos(w2 * t + self.phiy) * w2,
                -win * self.az * np.sin(w3 * t + self.phiz) * w3**2
                + 2 * d_win * self.az * np.cos(w3 * t + self.phiz) * w3,
            ]
        )
        da = np.array(
            [
                -win * self.ax * np.cos(w1 * t + self.phix) * w1**3
                - 3 * d_win * self.ax * np.sin(w1 * t + self.phix) * w1**2,
                -win * self.ay * np.cos(w2 * t + self.phiy) * w2**3
                - 3 * d_win * self.ay * np.sin(w2 * t + self.phiy) * w2**2,
                -win * self.az * np.cos(w3 * t + self.phiz) * w3**3
                - 3 * d_win * self.az * np.sin(w3 * t + self.phiz) * w3**2,
            ]
        )
        d2a = np.array(
            [
                win * self.ax * np.sin(w1 * t + self.phix) * w1**4
                - 4 * d_win * self.ax * np.cos(w1 * t + self.phix) * w1**3,
                win * self.ay * np.sin(w2 * t + self.phiy) * w2**4
                - 4 * d_win * self.ay * np.cos(w2 * t + self.phiy) * w2**3,
                win * self.az * np.sin(w3 * t + self.phiz) * w3**4
                - 4 * d_win * self.az * np.cos(w3 * t + self.phiz) * w3**3,
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

        return x, v, a, da, q, dq, d2q

    def get(self, t):
        """Return desired state at time t, maintaining hovering outside trajectory range"""
        if t < 5:
            return (
                self.compute(0)[0],
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
            )
        elif t > self._tf - 5:
            return (
                self.compute(self._tf - 5)[0],
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
            )
        return self.compute(t)
