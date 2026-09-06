"""Reference trajectories for cable-suspended payloads."""

import numpy as np

from roverfly.trajectories.base import Trajectory


class CrazyTrajectoryPayload(Trajectory):
    def __init__(self, tf=30, ax=2, ay=2, az=1, f1=0.2, f2=0.2, f3=0.1):
        super().__init__(tf)

        self.ax = ax
        self.ay = ay
        self.az = az

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

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

        self.mP = 0.1
        self.g = 9.81
        self.e3 = np.array([0, 0, 1])

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
        """Compute trajectory with smooth transition"""
        w1, w2, w3 = [2 * np.pi * f for f in (self.f1, self.f2, self.f3)]
        win = self.window(t)
        d_win = self.d_window(t)

        x = np.array(
            [
                win * self.ax * (1 - np.cos(w1 * t + self.phix)),
                win * self.ay * (1 - np.cos(w2 * t + self.phiy)),
                win * self.az * (1 - np.cos(w3 * t + self.phiz)),
            ]
        )
        v = np.array(
            [
                win * self.ax * np.sin(w1 * t + self.phix) * w1
                + d_win * self.ax * (1 - np.cos(w1 * t + self.phix)),
                win * self.ay * np.sin(w2 * t + self.phiy) * w2
                + d_win * self.ay * (1 - np.cos(w2 * t + self.phiy)),
                win * self.az * np.sin(w3 * t + self.phiz) * w3
                + d_win * self.az * (1 - np.cos(w3 * t + self.phiz)),
            ]
        )
        a = np.array(
            [
                win * self.ax * np.cos(w1 * t + self.phix) * w1 * w1
                + 2 * d_win * self.ax * np.sin(w1 * t + self.phix) * w1,
                win * self.ay * np.cos(w2 * t + self.phiy) * w2 * w2
                + 2 * d_win * self.ay * np.sin(w2 * t + self.phiy) * w2,
                win * self.az * np.cos(w3 * t + self.phiz) * w3 * w3
                + 2 * d_win * self.az * np.sin(w3 * t + self.phiz) * w3,
            ]
        )
        da = np.array(
            [
                -win * self.ax * np.sin(w1 * t + self.phix) * w1 * w1 * w1
                + 3 * d_win * self.ax * np.cos(w1 * t + self.phix) * w1 * w1,
                -win * self.ay * np.sin(w2 * t + self.phiy) * w2 * w2 * w2
                + 3 * d_win * self.ay * np.cos(w2 * t + self.phiy) * w2 * w2,
                -win * self.az * np.sin(w3 * t + self.phiz) * w3 * w3 * w3
                + 3 * d_win * self.az * np.cos(w3 * t + self.phiz) * w3 * w3,
            ]
        )
        d2a = np.array(
            [
                -win * self.ax * np.cos(w1 * t + self.phix) * w1 * w1 * w1 * w1
                - 4 * d_win * self.ax * np.sin(w1 * t + self.phix) * w1 * w1 * w1,
                -win * self.ay * np.cos(w2 * t + self.phiy) * w2 * w2 * w2 * w2
                - 4 * d_win * self.ay * np.sin(w2 * t + self.phiy) * w2 * w2 * w2,
                -win * self.az * np.cos(w3 * t + self.phiz) * w3 * w3 * w3 * w3
                - 4 * d_win * self.az * np.sin(w3 * t + self.phiz) * w3 * w3 * w3,
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


class CircularTrajPayload(Trajectory):
    def __init__(self, r=3, origin=np.zeros(3), w=2 * np.pi * 0.2, tf=10, accel_duration=2):
        super().__init__(tf)
        self.r = r
        self.origin = origin
        self.w = w
        self.accel_duration = self._tf / 2  # Duration of acceleration phase
        self.w_max = w  # Maximum angular velocity

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0, 0, 1])

    def get(self, t):
        if t < self.accel_duration:
            # Acceleration phase
            w_t = self.w_max * (t / self.accel_duration)
            w_dot = self.w_max / self.accel_duration
        else:
            # Constant velocity phase
            w_t = self.w_max
            w_dot = 0

        x = self.origin + self.r * np.array([np.cos(w_t * t), np.sin(w_t * t), 0])
        v = self.r * np.array([-w_t * np.sin(w_t * t), w_t * np.cos(w_t * t), 0])
        a = self.r * np.array([-(w_t**2) * np.cos(w_t * t), -(w_t**2) * np.sin(w_t * t), 0])

        da = self.r * np.array(
            [
                -2 * w_t * w_dot * np.cos(w_t * t) + w_t**3 * np.sin(w_t * t),
                -2 * w_t * w_dot * np.sin(w_t * t) - w_t**3 * np.cos(w_t * t),
                0,
            ]
        )

        d2a = self.r * np.array(
            [
                2 * w_dot**2 * np.cos(w_t * t)
                - 4 * w_t**2 * w_dot * np.sin(w_t * t)
                - w_t**4 * np.cos(w_t * t),
                2 * w_dot**2 * np.sin(w_t * t)
                + 4 * w_t**2 * w_dot * np.cos(w_t * t)
                - w_t**4 * np.sin(w_t * t),
                0,
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


class CrazyTrajectoryPayloadSwing(Trajectory):
    def __init__(self, tf=10, ax=5, ay=5, az=5, f1=0.5, f2=0.5, f3=0.5):
        super().__init__(tf)
        self.ax = ax
        self.ay = ay
        self.az = az
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        # self.phix = np.random.choice([np.pi/2, 3*np.pi/2])
        # self.phiy = np.random.choice([np.pi/2, 3*np.pi/2])
        # self.phiz = np.random.choice([np.pi/2, 3*np.pi/2])
        self.phix = 0
        self.phiy = 0
        self.phiz = 0

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0, 0, 1])

    def get(self, t):
        w1 = 2 * np.pi * self.f1
        w2 = 2 * np.pi * self.f2
        w3 = 2 * np.pi * self.f3

        x = np.array(
            [
                self.ax * (np.sin(w1 * t + self.phix)),
                self.ay * (1 - np.cos(w2 * t + self.phiy)),
                self.az * (-1 - np.cos(w3 * t + self.phiz)),
            ]
        )
        v = np.array(
            [
                self.ax * np.cos(w1 * t + self.phix) * w1,
                self.ay * np.sin(w2 * t + self.phiy) * w2,
                self.az * np.sin(w3 * t + self.phiz) * w3,
            ]
        )
        a = np.array(
            [
                -self.ax * np.sin(w1 * t + self.phix) * w1 * w1,
                self.ay * np.cos(w2 * t + self.phiy) * w2 * w2,
                self.az * np.cos(w3 * t + self.phiz) * w3 * w3,
            ]
        )
        da = np.array(
            [
                -self.ax * np.cos(w1 * t + self.phix) * w1 * w1 * w1,
                -self.ay * np.sin(w2 * t + self.phiy) * w2 * w2 * w2,
                -self.az * np.sin(w3 * t + self.phiz) * w3 * w3 * w3,
            ]
        )
        d2a = np.array(
            [
                self.ax * np.sin(w1 * t + self.phix) * w1 * w1 * w1 * w1,
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

        return x, v, a, da, q, dq, d2q


class CustomTrajectoryPayloadWindow(Trajectory):
    def __init__(self, tf=4):
        super().__init__(tf)
        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0, 0, 1])
        self.L_x = 4.0  # Final value for x sigmoid
        self.L_z = 1.0  # Final value for z double sigmoid
        self.k_x = 2.5  # Growth rate for x sigmoid (Fixed)
        self.k_z = 3.0  # Growth rate for z double sigmoid (Fixed)
        self.t0_x = 1.5  # Center of x sigmoid (Fixed)
        self.t0_z = 1.5  # Center of z double sigmoid (Fixed)
        self.d_z = 2.0  # Shift for second sigmoid in z (Fixed)

    def sigmoid(self, t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))

    def sigmoid_derivative(self, t, L, k, t0):
        exp_term = np.exp(-k * (t - t0))
        return (L * k * exp_term) / (1 + exp_term) ** 2

    def sigmoid_second_derivative(self, t, L, k, t0):
        exp_term = np.exp(-k * (t - t0))
        return L * k**2 * exp_term * (1 + exp_term) ** (-2) * (-1 + 2 * exp_term / (1 + exp_term))

    def double_sigmoid_decrease(self, t, L, k, t0, d):
        sigmoid_1 = self.sigmoid(t, L, k, t0)
        sigmoid_2 = self.sigmoid(t, L, k, t0 + d)
        return -(sigmoid_1 + sigmoid_2)

    def double_sigmoid_decrease_derivative(self, t, L, k, t0, d):
        sigmoid_derivative_1 = self.sigmoid_derivative(t, L, k, t0)
        sigmoid_derivative_2 = self.sigmoid_derivative(t, L, k, t0 + d)
        return -(sigmoid_derivative_1 + sigmoid_derivative_2)

    def double_sigmoid_decrease_second_derivative(self, t, L, k, t0, d):
        sigmoid_second_derivative_1 = self.sigmoid_second_derivative(t, L, k, t0)
        sigmoid_second_derivative_2 = self.sigmoid_second_derivative(t, L, k, t0 + d)
        return -(sigmoid_second_derivative_1 + sigmoid_second_derivative_2)

    def get(self, t):
        if t <= self._tf:
            x = np.array(
                [
                    self.sigmoid(t, self.L_x, self.k_x, self.t0_x),
                    0,
                    self.double_sigmoid_decrease(t, self.L_z, self.k_z, self.t0_z, self.d_z),
                ]
            )
            v = np.array(
                [
                    self.sigmoid_derivative(t, self.L_x, self.k_x, self.t0_x),
                    0,
                    self.double_sigmoid_decrease_derivative(
                        t, self.L_z, self.k_z, self.t0_z, self.d_z
                    ),
                ]
            )
            a = np.array(
                [
                    self.sigmoid_second_derivative(t, self.L_x, self.k_x, self.t0_x),
                    0,
                    self.double_sigmoid_decrease_second_derivative(
                        t, self.L_z, self.k_z, self.t0_z, self.d_z
                    ),
                ]
            )
        else:
            dt = t - self._tf
            x_tf = np.array(
                [
                    self.sigmoid(self._tf, self.L_x, self.k_x, self.t0_x),
                    0,
                    self.double_sigmoid_decrease(self._tf, self.L_z, self.k_z, self.t0_z, self.d_z),
                ]
            )
            v_tf = np.array(
                [
                    self.sigmoid_derivative(self._tf, self.L_x, self.k_x, self.t0_x),
                    0,
                    self.double_sigmoid_decrease_derivative(
                        self._tf, self.L_z, self.k_z, self.t0_z, self.d_z
                    ),
                ]
            )
            a_tf = np.array(
                [
                    self.sigmoid_second_derivative(self._tf, self.L_x, self.k_x, self.t0_x),
                    0,
                    self.double_sigmoid_decrease_second_derivative(
                        self._tf, self.L_z, self.k_z, self.t0_z, self.d_z
                    ),
                ]
            )
            x = x_tf + v_tf * dt + 0.5 * a_tf * dt**2
            v = v_tf + a_tf * dt
            a = a_tf

        da = np.gradient(a, np.diff(t).mean() if np.size(t) > 1 else 1)
        d2a = np.gradient(da, np.diff(t).mean() if np.size(t) > 1 else 1)

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
