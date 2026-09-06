"""Reference trajectories for a single quadrotor."""

import numpy as np

from roverfly.trajectories.base import QuinticTrajectory, Trajectory


class CircularTraj(Trajectory):
    def __init__(self, r=3, origin=np.zeros(3), w=2 * np.pi * 0.05, tf=100, accel_duration=10):
        super().__init__(tf)
        self.r = r
        self.origin = origin
        self.w = w
        self.accel_duration = self._tf / 4  # Duration of acceleration phase
        self.w_max = w  # Maximum angular velocity

    def get(self, t):
        if t < self.accel_duration:
            # Acceleration phase
            w_t = self.w_max * (t / self.accel_duration)
        else:
            # Constant velocity phase
            w_t = self.w_max

        x = self.origin + self.r * np.array([np.cos(w_t * t), np.sin(w_t * t), 0])
        v = self.r * np.array([-w_t * np.sin(w_t * t), w_t * np.cos(w_t * t), 0])
        a = self.r * np.array([-(w_t**2) * np.cos(w_t * t), -(w_t**2) * np.sin(w_t * t), 0])

        return x, v, a


class CrazyTrajectory(Trajectory):
    def __init__(self, tf=30, ax=2, ay=2, az=1, f1=0.2, f2=0.2, f3=0.1):
        super().__init__(tf)

        self.ax = np.random.uniform(ax / 2, ax)
        self.ay = np.random.uniform(ay / 2, ay)
        self.az = np.random.uniform(az / 2, az)

        self.f1 = np.random.uniform(f1 / 2, f1)
        self.f2 = np.random.uniform(f2 / 2, f2)
        self.f3 = np.random.uniform(f3 / 2, f3)

        self.phix = np.random.choice([np.pi / 2, 3 * np.pi / 2])
        self.phiy = np.random.choice([np.pi / 2, 3 * np.pi / 2])
        self.phiz = np.random.choice([np.pi / 2, 3 * np.pi / 2])

        self.v_max = 4.0
        self.w1, self.w2, self.w3 = [2 * np.pi * f for f in (self.f1, self.f2, self.f3)]

        if max(self.ax * self.w1, self.ay * self.w2, self.az * self.w3) > self.v_max:
            scaling_factor = (
                max(self.ax * self.w1, self.ay * self.w2, self.az * self.w3) / self.v_max
            )
            self.ax *= scaling_factor
            self.ay *= scaling_factor
            self.az *= scaling_factor

    def __str__(self):
        return (
            f"CrazyTrajectory:\n"
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
        """Compute position, velocity, and acceleration at time t"""
        win = self.window(t)
        d_win = self.d_window(t)

        x = np.array(
            [
                win * self.ax * (1 - np.cos(self.w1 * t + self.phix)),
                win * self.ay * (1 - np.cos(self.w2 * t + self.phiy)),
                win * self.az * (1 - np.cos(self.w3 * t + self.phiz)),
            ]
        )
        v = np.array(
            [
                win * self.ax * np.sin(self.w1 * t + self.phix) * self.w1
                + d_win * self.ax * (1 - np.cos(self.w1 * t + self.phix)),
                win * self.ay * np.sin(self.w2 * t + self.phiy) * self.w2
                + d_win * self.ay * (1 - np.cos(self.w2 * t + self.phiy)),
                win * self.az * np.sin(self.w3 * t + self.phiz) * self.w3
                + d_win * self.az * (1 - np.cos(self.w3 * t + self.phiz)),
            ]
        )
        a = np.array(
            [
                win * self.ax * np.cos(self.w1 * t + self.phix) * self.w1 * self.w1
                + 2 * d_win * self.ax * np.sin(self.w1 * t + self.phix) * self.w1,
                win * self.ay * np.cos(self.w2 * t + self.phiy) * self.w2 * self.w2
                + 2 * d_win * self.ay * np.sin(self.w2 * t + self.phiy) * self.w2,
                win * self.az * np.cos(self.w3 * t + self.phiz) * self.w3 * self.w3
                + 2 * d_win * self.az * np.sin(self.w3 * t + self.phiz) * self.w3,
            ]
        )

        return x, v, a

    def get(self, t):
        """Return the desired state at time t, maintaining hovering outside the trajectory range"""
        if t < 5:
            return (
                self.compute(0)[0],
                np.zeros(3),
                np.zeros(3),
            )  # Maintain hovering at the initial position
        elif t > self._tf - 5:
            return (
                self.compute(self._tf - 5)[0],
                np.zeros(3),
                np.zeros(3),
            )  # Maintain hovering at the final position
        return self.compute(t)


class FullCrazyTrajectory(Trajectory):
    def __init__(self, traj, tf=40):
        """
        Full trajectory that includes:
        1. Smooth takeoff (0-5s) using a QuinticTrajectory.
        2. Crazy trajectory (5-35s).
        3. Smooth landing (35-40s) using a QuinticTrajectory.

        Args:
            traj (Trajectory): An instance of CrazyTrajectory.
            tf (float): Total duration of the trajectory (default: 40s).
        """
        super().__init__(tf)
        self.crazy_traj = traj
        self.takeoff_height = 1.5  # Target height for takeoff and landing

        # Define takeoff and landing trajectories using QuinticTrajectory
        self.takeoff_traj = QuinticTrajectory(
            tf=5, x0=np.array([0, 0, 0]), xf=np.array([0, 0, self.takeoff_height])
        )
        self.landing_traj = QuinticTrajectory(
            tf=5, x0=np.array([0, 0, self.takeoff_height]), xf=np.array([0, 0, 0])
        )

    def get(self, t):
        if t < 5:
            return self.takeoff_traj.get(t)  # Takeoff phase
        elif t < self._tf - 5:
            crazy_x, crazy_v, crazy_a = self.crazy_traj.get(t - 5)
            crazy_x[2] += self.takeoff_height  # Shift trajectory to hover at 1.5m height
            return crazy_x, crazy_v, crazy_a
        else:
            return self.landing_traj.get(t - (self._tf - 5))
