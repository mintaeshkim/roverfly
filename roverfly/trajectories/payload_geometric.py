"""Geometric and figure-eight payload trajectories."""

import warnings

import numpy as np

from roverfly.trajectories.base import Trajectory


class GeometricTrajectoryPayload(Trajectory):
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
                self.ax * np.cos(w1 * t + self.phix) * w1**2,
                self.ay * np.cos(w2 * t + self.phiy) * w2**2,
                self.az * np.cos(w3 * t + self.phiz) * w3**2,
            ]
        )
        da = np.array(
            [
                -self.ax * np.sin(w1 * t + self.phix) * w1**3,
                -self.ay * np.sin(w2 * t + self.phiy) * w2**3,
                -self.az * np.sin(w3 * t + self.phiz) * w3**3,
            ]
        )
        d2a = np.array(
            [
                -self.ax * np.cos(w1 * t + self.phix) * w1**4,
                -self.ay * np.cos(w2 * t + self.phiy) * w2**4,
                -self.az * np.cos(w3 * t + self.phiz) * w3**4,
            ]
        )
        d3a = np.array(
            [
                self.ax * np.sin(w1 * t + self.phix) * w1**5,
                self.ay * np.sin(w2 * t + self.phiy) * w2**5,
                self.az * np.sin(w3 * t + self.phiz) * w3**5,
            ]
        )
        d4a = np.array(
            [
                -self.ax * np.cos(w1 * t + self.phix) * w1**6,
                -self.ay * np.cos(w2 * t + self.phiy) * w2**6,
                -self.az * np.cos(w3 * t + self.phiz) * w3**6,
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

        d3Tp = -self.mP * d3a
        d3norm_Tp = (
            2 * np.dot(d2Tp, dTp) + np.dot(dTp, d2Tp) + np.dot(Tp, d3Tp) - 3 * dnorm_Tp * d2norm_Tp
        ) / norm_Tp
        d3q = (
            d3Tp
            - d2q * dnorm_Tp
            - dq * d2norm_Tp
            - dq * d2norm_Tp
            - q * d3norm_Tp
            - d2q * dnorm_Tp
            - dq * d2norm_Tp
            - d2q * dnorm_Tp
        ) / norm_Tp

        d4Tp = -self.mP * d4a
        d4norm_Tp = (
            2 * np.dot(d3Tp, dTp)
            + 2 * np.dot(d2Tp, d2Tp)
            + np.dot(d2Tp, d2Tp)
            + np.dot(dTp, d3Tp)
            + np.dot(dTp, d3Tp)
            + np.dot(Tp, d4Tp)
            - 3 * d2norm_Tp**2
            - 3 * dnorm_Tp * d3norm_Tp
            - d3norm_Tp * dnorm_Tp
        ) / norm_Tp
        d4q = (
            d4Tp
            - d3q * dnorm_Tp
            - d2q * d2norm_Tp
            - d2q * d2norm_Tp
            - dq * d3norm_Tp
            - d2q * d2norm_Tp
            - dq * d3norm_Tp
            - dq * d3norm_Tp
            - q * d4norm_Tp
            - d3q * dnorm_Tp
            - d2q * d2norm_Tp
            - d2q * d2norm_Tp
            - dq * d3norm_Tp
            - d3q * dnorm_Tp
            - d2q * d2norm_Tp
            - d3q * dnorm_Tp
        ) / norm_Tp

        return x, v, a, da, d2a, d3a, d4a, q, dq, d2q, d3q, d4q


class PayloadFigureEight(Trajectory):
    """
    Payload figure-8 (Lissajous) trajectory with altitude modulation.
    Returns (x, v, a, da, q, dq, d2q) to match CrazyTrajectoryPayload interface.
    """

    def __init__(
        self,
        tf=60.0,
        R=2.0,  # horizontal radius [m]
        Az=0.75,  # altitude modulation amplitude [m]
        z0=1.5,  # nominal altitude [m]
        omega=0.5,  # horizontal base frequency [rad/s]
        omegaz=0.25,  # vertical frequency [rad/s]
        mP=0.2,  # payload mass [kg]
        v_max=3.0,  # horizontal speed cap for safety
        g=9.81,
    ):
        super().__init__(tf)
        self.R = float(R)
        self.Az = float(Az)
        self.z0 = float(z0)
        self.omega = float(omega)
        self.omegaz = float(omegaz)
        self.mP = float(mP)
        self.v_max = float(v_max)
        self.g = float(g)
        self.e3 = np.array([0.0, 0.0, 1.0], dtype=float)

        # Optional quick safety check on nominal peak horizontal speed
        vxy_peak = self.R * self.omega * np.sqrt(1.0**2 + 2.0**2)  # ~R*omega*sqrt(5)
        if vxy_peak > self.v_max:
            scale = vxy_peak / self.v_max
            self.R /= scale
            warnings.warn(
                f"[PayloadFigureEight] R scaled down by {scale:.2f} to respect v_max≈{self.v_max} m/s."
            )

    # --- smooth start/stop window (same shape as your CrazyTrajectoryPayload) ---
    def window(self, t):
        t_start, t_end = 5.0, self._tf - 5.0
        T = 3.0  # transition duration
        if t < t_start or t > t_end:
            return 0.0
        if t_start <= t < t_start + T:
            x = (t - t_start) / T
            return 3 * x**2 - 2 * x**3
        if t_end - T < t <= t_end:
            x = (t_end - t) / T
            return 3 * x**2 - 2 * x**3
        return 1.0

    def d_window(self, t):
        t_start, t_end = 5.0, self._tf - 5.0
        T = 3.0
        if t_start <= t < t_start + T:
            x = (t - t_start) / T
            return (6 * x - 6 * x**2) / T
        if t_end - T < t <= t_end:
            x = (t_end - t) / T
            return (-6 * x + 6 * x**2) / T
        return 0.0

    def compute(self, t):
        """
        Lissajous figure-8 in XY:
            x =  R cos(omega t)
            y =  R sin(2 omega t)
        Altitude modulation:
            z = z0 + win * Az sin(omegaz t)

        We multiply horizontal components by win to ensure hover before/after.
        Derivatives include product-rule terms with win' (d_win).
        """
        w = self.omega
        wz = self.omegaz
        win = self.window(t)
        d_win = self.d_window(t)

        # --- position ---
        cx = np.cos(w * t)
        sx = np.sin(w * t)
        cy2 = np.cos(2 * w * t)
        sy2 = np.sin(2 * w * t)
        sz = np.sin(wz * t)
        cz = np.cos(wz * t)

        x = np.array(
            [win * self.R * cx, win * self.R * sy2, self.z0 + win * self.Az * sz], dtype=float
        )

        # --- velocity (product rule with win, d_win) ---
        dx = np.array(
            [
                -win * self.R * w * sx + d_win * self.R * cx,
                win * self.R * 2 * w * cy2 + d_win * self.R * sy2,
                win * self.Az * wz * cz + d_win * self.Az * sz,
            ],
            dtype=float,
        )

        # --- acceleration ---
        ddx = np.array(
            [
                -win * self.R * w**2 * cx
                + 2 * d_win * (-self.R * w * sx)
                + 0.0 * self.R * cx,  # d2(win*R*cx)
                -win * self.R * (2 * w) ** 2 * sy2 + 2 * d_win * (self.R * 2 * w * cy2) + 0.0,
                -win * self.Az * wz**2 * sz + 2 * d_win * (self.Az * wz * cz) + 0.0,
            ],
            dtype=float,
        )
        # 위에서 마지막 0.0 항은 d2_win * (base)인데, CrazyTrajectoryPayload에 맞춰 2* d_win * (first-derivative)까지만 사용.
        # 필요하다면 d2_window까지 포함하도록 확장 가능.

        # --- jerk (third derivative) ---
        # For stability, we keep a consistent structure with CrazyTrajectoryPayload: approximate 'da' as time-derivative of acceleration
        # ignoring second derivative of window (like the provided code's style).
        # Compute symbolic derivatives of the inner terms:
        # Derivatives we need explicitly:
        d_term_x = self.R * w**3 * sx  # d/dt[-R w^2 cos wt] =  R w^3 sin wt
        d_term_y = -self.R * (2 * w) ** 3 * cy2
        d_term_z = -self.Az * wz**3 * cz

        # derivative of first-derivatives (inside the 2*d_win*(...)):
        v_inner_x = -self.R * w * sx
        v_inner_y = self.R * 2 * w * cy2
        v_inner_z = self.Az * wz * cz

        # jerk (approximate, matching CrazyTrajectoryPayload style: da)
        da = np.array(
            [
                -win * d_term_x + 2 * d_win * v_inner_x,
                win * d_term_y + 2 * d_win * v_inner_y,
                win * d_term_z + 2 * d_win * v_inner_z,
            ],
            dtype=float,
        )

        # --- snap (fourth derivative) as 'd2a' in the same style (again approximate form consistent with given code) ---
        d2_term_x = self.R * w**4 * cx  # d/dt of d_term_x
        d2_term_y = self.R * (2 * w) ** 4 * sy2  # note sign from cosine derivative
        d2_term_z = self.Az * wz**4 * sz

        d2a = np.array(
            [
                -win * d2_term_x
                - 4
                * d_win
                * d_term_x
                / (w if w != 0 else 1.0),  # keep same pattern depth as reference
                -win * d2_term_y - 4 * d_win * d_term_y / (2 * w if w != 0 else 1.0),
                -win * d2_term_z - 4 * d_win * d_term_z / (wz if wz != 0 else 1.0),
            ],
            dtype=float,
        )

        # --- cable direction q from payload tension Tp = -mP (a + g e3) ---
        Tp = -self.mP * (ddx + self.g * self.e3)
        norm_Tp = np.linalg.norm(Tp)
        eps = 1e-9
        if norm_Tp < eps:
            warnings.warn(
                "[PayloadFigureEight] |Tp| near zero; clamping to avoid numerical blow-up."
            )
            norm_Tp = eps
        q = Tp / norm_Tp

        # time derivatives of Tp for dq, d2q (using da, d2a)
        dTp = -self.mP * da
        dnorm_Tp = (Tp @ dTp) / norm_Tp
        dq = (dTp - q * dnorm_Tp) / norm_Tp

        d2Tp = -self.mP * d2a
        d2norm_Tp = ((dTp @ dTp) + (Tp @ d2Tp) - dnorm_Tp**2) / norm_Tp
        d2q = (d2Tp - dq * dnorm_Tp - q * d2norm_Tp - dq * dnorm_Tp) / norm_Tp

        # position / velocity / acceleration outputs follow CrazyTrajectoryPayload convention:
        # x (payload pos), v, a, da (jerk), q, dq, d2q
        return x, dx, ddx, da, q, dq, d2q

    def get(self, t):
        """
        Maintain hovering (zero motion) outside the active window,
        matching CrazyTrajectoryPayload's interface (returns 7 vectors).
        """
        t0 = 5.0
        t1 = self._tf - 5.0
        if t < t0:
            x0 = np.array([0.0, 0.0, self.z0])
            z = np.zeros(3)
            return x0, z, z, z, np.array([0.0, 0.0, 1.0]), z, z
        if t > t1:
            x1, _, _, _, _, _, _ = self.compute(t1)
            z = np.zeros(3)
            return x1, z, z, z, np.array([0.0, 0.0, 1.0]), z, z
        return self.compute(t)
