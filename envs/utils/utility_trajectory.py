# Helpers
import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'utils'))
import numpy as np
import warnings
import matplotlib.pyplot as plt
from utils.geo_tools import *

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

        axs[0].plot(T, x[:, 0], 'r', linewidth=2, label='x')
        axs[0].plot(T, x[:, 1], 'g', linewidth=2, label='y')
        axs[0].plot(T, x[:, 2], 'b', linewidth=2, label='z')
        axs[0].set_title('Position')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(T, v[:, 0], ':r', linewidth=2, label='vx')
        axs[1].plot(T, v[:, 1], ':g', linewidth=2, label='vy')
        axs[1].plot(T, v[:, 2], ':b', linewidth=2, label='vz')
        axs[1].set_title('Velocity')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(T, a[:, 0], '--r', linewidth=2, label='ax')
        axs[2].plot(T, a[:, 1], '--g', linewidth=2, label='ay')
        axs[2].plot(T, a[:, 2], '--b', linewidth=2, label='az')
        axs[2].set_title('Acceleration')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
    
    def plot3d(self):
        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        for t in T:
            x_, _, _ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory', color='b', linewidth=2)
        ax.set_title('3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def plot3d_payload(self):
        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        q = np.empty((0, 3))
        for t in T:
            x_, _, _, _, q_, _, _ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)
            q = np.append(q, np.array([q_]), axis=0)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory', color='b', linewidth=2)
        ax.set_title('3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        max_range = np.array([x[:, 0].max()-x[:, 0].min(), x[:, 1].max()-x[:, 1].min(), x[:, 2].max()-x[:, 2].min()]).max() / 2.0
        mid_x = (x[:, 0].max()+x[:, 0].min()) * 0.5
        mid_y = (x[:, 1].max()+x[:, 1].min()) * 0.5
        mid_z = (x[:, 2].max()+x[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        step = len(T) // 20
        for i in range(0, len(T), step):
            ax.quiver(x[i, 0], x[i, 1], x[i, 2], -q[i, 0], -q[i, 1], -q[i, 2], length=0.5, normalize=True, color='r',  arrow_length_ratio=0.01)

        plt.show()
    
    def plot3d_payload_geometric(self):
        T = np.linspace(0, self._tf, 100)

        x = np.empty((0, 3))
        q = np.empty((0, 3))
        for t in T:
            x_, _, _, _, _, _, _, q_, _, _, _, _ = self.get(t)
            x = np.append(x, np.array([x_]), axis=0)
            q = np.append(q, np.array([q_]), axis=0)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory', color='b', linewidth=2)
        ax.set_title('3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        max_range = np.array([x[:, 0].max()-x[:, 0].min(), x[:, 1].max()-x[:, 1].min(), x[:, 2].max()-x[:, 2].min()]).max() / 2.0
        mid_x = (x[:, 0].max()+x[:, 0].min()) * 0.5
        mid_y = (x[:, 1].max()+x[:, 1].min()) * 0.5
        mid_z = (x[:, 2].max()+x[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        step = len(T) // 20
        for i in range(0, len(T), step):
            ax.quiver(x[i, 0], x[i, 1], x[i, 2], -q[i, 0], -q[i, 1], -q[i, 2], length=0.5, normalize=True, color='r',  arrow_length_ratio=0.01)

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

        self._t = lambda l: np.array([1., l, l**2, l**3, l**4, l**5])

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
            l = t / self._tf
            return (np.array([self._t(l)]) @ self._pos_params)[0],\
                   (np.array([self._t(l)]) @ self._vel_params)[0],\
                   (np.array([self._t(l)]) @ self._acc_params)[0]


class SmoothTraj5(SmoothTraj):
    def compute_traj_params(self):
        a = self._xf - self._x0
        self._pos_params = np.array([self._x0, np.zeros(3), np.zeros(3), 10*a, -15*a, 6*a])
        self._vel_params = np.array([np.zeros(3), 2*np.zeros(3), 3*10*a, 4*(-15)*a, 5*6*a, np.zeros(3)])
        self._acc_params = np.array([np.zeros(3), 6*10*a, 12*(-15)*a, 20*6*a, np.zeros(3), np.zeros(3)])


class SmoothTraj3(SmoothTraj):
    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda l: np.array([1., l, l**2, l**3])

    def compute_traj_params(self):
        a = self._xf - self._x0
        self._pos_params = np.array([self._x0, np.zeros(3), 3*a, -2*a])
        self._vel_params = np.array([np.zeros(3), 6*a, -6*a, np.zeros(3)])
        self._acc_params = np.array([6*a, -12*a, np.zeros(3), np.zeros(3)])


class SmoothTraj1(SmoothTraj):
    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda l: np.array([1., l])

    def compute_traj_params(self):
        a = self._xf - self._x0
        self._pos_params = np.array([self._x0, a])
        self._vel_params = np.array([a, np.zeros(3)])
        self._acc_params = np.array([np.zeros(3), np.zeros(3)])


class CircularTraj(Trajectory):
    def __init__(self, r=3, origin=np.zeros(3), w=2*np.pi*0.05, tf=100, accel_duration=10):
        super().__init__(tf)
        self.r = r
        self.origin = origin
        self.w = w
        self.accel_duration = self._tf/4  # Duration of acceleration phase
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
        a = self.r * np.array([-w_t**2 * np.cos(w_t * t), -w_t**2 * np.sin(w_t * t), 0])

        return x, v, a


class SmoothSineTraj(SmoothTraj):
    def compute_traj_params(self):
        self._pos_offset = 0.5 * (self._xf + self._x0)
        self._pos_amp = 0.5 * (self._xf - self._x0)
        self._vel_amp = 0.5 * (self._xf - self._x0) * (np.pi / self._tf)
        self._acc_amp = -0.5 * (self._xf - self._x0) * (np.pi / self._tf)**2

    def get(self, t):
        if t >= self._tf:
            return self._xf, np.zeros(3), np.zeros(3)
        elif t <= 0:
            return self._x0, np.zeros(3), np.zeros(3)
        else:
            x = self._pos_offset + self._pos_amp * np.sin(t * np.pi / self._tf - np.pi / 2)
            v = self._vel_amp * np.cos(t * np.pi / self._tf - np.pi / 2)
            a = self._acc_amp * np.sin(t * np.pi / self._tf - np.pi / 2)
            return x, v, a


class CrazyTrajectory(Trajectory):
    def __init__(self, tf=30, ax=5, ay=5, az=5, f1=0.5, f2=0.5, f3=0.5):
        super().__init__(tf)
        alpha, beta = 5.0, 5.0
        self.ax, self.ay, self.az = [a * np.random.beta(alpha, beta) for a in (ax, ay, az)]
        self.f1, self.f2, self.f3 = [f * np.random.beta(alpha, beta) for f in (f1, f2, f3)]
        self.phix, self.phiy, self.phiz = np.random.choice([0, np.pi], size=3)

    def window(self, t):
        """Cosine window function for smooth velocity transitions at t=5s and t=25s"""
        if t < 5 or t > 25:
            return 0  # Hovering state (no movement)
        elif 5 <= t < 7.5:
            return 0.5 * (1 - np.cos(np.pi * (t - 5) / 2.5))  # Smooth transition start
        elif 22.5 < t <= 25:
            return 0.5 * (1 - np.cos(np.pi * (25 - t) / 2.5))  # Smooth transition end
        return 1  # Full trajectory motion

    def d_window(self, t):
        """Derivative of the window function for velocity adjustment"""
        if 5 <= t < 7.5:
            return 0.5 * (np.pi / 2.5) * np.sin(np.pi * (t - 5) / 2.5)
        elif 22.5 < t <= 25:
            return -0.5 * (np.pi / 2.5) * np.sin(np.pi * (25 - t) / 2.5)
        return 0  # No change in hovering state

    def compute(self, t):
        """Compute position, velocity, and acceleration at time t"""
        w = [2 * np.pi * f for f in (self.f1, self.f2, self.f3)]  # Convert frequencies to angular velocities
        phases = [self.phix, self.phiy, self.phiz]
        win = self.window(t)
        d_win = self.d_window(t)

        # Compute position
        x = np.array([win * a * np.sin(wi * t + phi) for a, wi, phi in zip((self.ax, self.ay, self.az), w, phases)])
        # Compute velocity with window function scaling
        v = np.array([
            win * a * np.cos(wi * t + phi) * wi + d_win * a * np.sin(wi * t + phi)
            for a, wi, phi in zip((self.ax, self.ay, self.az), w, phases)
        ])
        # Compute acceleration with second derivative adjustments
        a = np.array([
            win * (-a * np.sin(wi * t + phi) * wi**2) + 2 * d_win * a * np.cos(wi * t + phi) * wi
            for a, wi, phi in zip((self.ax, self.ay, self.az), w, phases)
        ])
        
        return x, v, a

    def get(self, t):
        """Return the desired state at time t, maintaining hovering outside the trajectory range"""
        if t < 5:
            return self.compute(0)[0], np.zeros(3), np.zeros(3)  # Maintain hovering at the initial position
        elif t > 25:
            return self.compute(25)[0], np.zeros(3), np.zeros(3)  # Maintain hovering at the final position
        return self.compute(t)

        
class SmoothTrajClipped(Trajectory):
    def __init__(self, x0, xf, max_velocity=5.0, max_acceleration=10.0):
        self._x0 = np.array(x0)
        self._xf = np.array(xf)
        self._max_velocity = max_velocity  # m/s
        self._max_acceleration = max_acceleration  # m/s^2

        self._tf = self._compute_time()
        self._compute_traj_params()

    def _compute_time(self):
        # Initial guess for tf
        tf = 5.0
        while True:
            self._tf = tf
            self._compute_traj_params()
            max_vel, max_acc = self._compute_max_vel_acc()
            if max_vel <= self._max_velocity and max_acc <= self._max_acceleration:
                break
            tf *= 1.1  # Increase tf to reduce velocity and acceleration
        return tf

    def _compute_max_vel_acc(self):
        t_values = np.linspace(0, self._tf, 100)
        max_vel = 0
        max_acc = 0
        for t in t_values:
            _, v, a = self.get(t)
            max_vel = max(max_vel, np.linalg.norm(v))
            max_acc = max(max_acc, np.linalg.norm(a))
        return max_vel, max_acc

    def _compute_traj_params(self):
        # Compute cubic polynomial coefficients
        self._a0 = self._x0
        self._a1 = np.zeros(3)
        self._a2 = 3 * (self._xf - self._x0) / (self._tf ** 2)
        self._a3 = -2 * (self._xf - self._x0) / (self._tf ** 3)

    def get(self, t):
        if t < 0:
            return self._x0, np.zeros(3), np.zeros(3)
        elif t > self._tf:
            return self._xf, np.zeros(3), np.zeros(3)
        else:
            x = self._a0 + self._a1 * t + self._a2 * t ** 2 + self._a3 * t ** 3
            v = self._a1 + 2 * self._a2 * t + 3 * self._a3 * t ** 2
            a = 2 * self._a2 + 6 * self._a3 * t
            return x, v, a


class CrazyTrajectoryPayload(Trajectory):
    def __init__(self, tf=10, ax=5, ay=5, az=5, f1=0.5, f2=0.5, f3=0.5):
        super().__init__(tf)
        alpha_param, beta_param = 5.0, 5.0
        self.ax = ax * np.random.beta(alpha_param, beta_param)
        self.ay = ay * np.random.beta(alpha_param, beta_param)
        self.az = az * np.random.beta(alpha_param, beta_param)
        
        self.f1 = f1 * np.random.beta(alpha_param, beta_param)
        self.f2 = f2 * np.random.beta(alpha_param, beta_param)
        self.f3 = f3 * np.random.beta(alpha_param, beta_param)

        self.phix = np.random.choice([np.pi/2, 3*np.pi/2])
        self.phiy = np.random.choice([np.pi/2, 3*np.pi/2])
        self.phiz = np.random.choice([np.pi/2, 3*np.pi/2])

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0,0,1])

    def get(self, t):
        w1 = 2 * np.pi * self.f1
        w2 = 2 * np.pi * self.f2
        w3 = 2 * np.pi * self.f3

        x = np.array([
            self.ax * (1 - np.cos(w1 * t + self.phix)),
            self.ay * (1 - np.cos(w2 * t + self.phiy)),
            self.az * (1 - np.cos(w3 * t + self.phiz))
        ])
        v = np.array([
            self.ax * np.sin(w1 * t + self.phix) * w1,
            self.ay * np.sin(w2 * t + self.phiy) * w2,
            self.az * np.sin(w3 * t + self.phiz) * w3
        ])
        a = np.array([
            self.ax * np.cos(w1 * t + self.phix) * w1 * w1,
            self.ay * np.cos(w2 * t + self.phiy) * w2 * w2,
            self.az * np.cos(w3 * t + self.phiz) * w3 * w3
        ])
        da = np.array([
            -self.ax * np.sin(w1 * t + self.phix) * w1 * w1 * w1,
            -self.ay * np.sin(w2 * t + self.phiy) * w2 * w2 * w2,
            -self.az * np.sin(w3 * t + self.phiz) * w3 * w3 * w3
        ])
        d2a = np.array([
            -self.ax * np.cos(w1 * t + self.phix) * w1 * w1 * w1 * w1,
            -self.ay * np.cos(w2 * t + self.phiy) * w2 * w2 * w2 * w2,
            -self.az * np.cos(w3 * t + self.phiz) * w3 * w3 * w3 * w3
        ])
        
        Tp = -self.mP * (a + self.g * self.e3)
        norm_Tp = np.linalg.norm(Tp)
        q = Tp / norm_Tp

        dTp = -self.mP * da
        dnorm_Tp = 1 / norm_Tp * np.dot(Tp, dTp)
        dq = (dTp - q * dnorm_Tp) / norm_Tp

        d2Tp = -self.mP * d2a ;
        d2norm_Tp = (np.dot(dTp, dTp) + np.dot(Tp, d2Tp) - dnorm_Tp**2) / norm_Tp
        d2q = (d2Tp - dq * dnorm_Tp - q * d2norm_Tp - dq * dnorm_Tp) / norm_Tp

        return x, v, a, da, q, dq, d2q


class CircularTrajPayload(Trajectory):
    def __init__(self, r=3, origin=np.zeros(3), w=2*np.pi*0.2, tf=10, accel_duration=2):
        super().__init__(tf)
        self.r = r
        self.origin = origin
        self.w = w
        self.accel_duration = self._tf/2  # Duration of acceleration phase
        self.w_max = w  # Maximum angular velocity

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0,0,1])

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
        a = self.r * np.array([-w_t**2 * np.cos(w_t * t), -w_t**2 * np.sin(w_t * t), 0])

        da = self.r * np.array([
            -2 * w_t * w_dot * np.cos(w_t * t) + w_t**3 * np.sin(w_t * t),
            -2 * w_t * w_dot * np.sin(w_t * t) - w_t**3 * np.cos(w_t * t),
            0
        ])

        d2a = self.r * np.array([
            2 * w_dot**2 * np.cos(w_t * t) - 4 * w_t**2 * w_dot * np.sin(w_t * t) - w_t**4 * np.cos(w_t * t),
            2 * w_dot**2 * np.sin(w_t * t) + 4 * w_t**2 * w_dot * np.cos(w_t * t) - w_t**4 * np.sin(w_t * t),
            0
        ])

        Tp = -self.mP * (a + self.g * self.e3)
        norm_Tp = np.linalg.norm(Tp)
        q = Tp / norm_Tp

        dTp = -self.mP * da
        dnorm_Tp = 1 / norm_Tp * np.dot(Tp, dTp)
        dq = (dTp - q * dnorm_Tp) / norm_Tp

        d2Tp = -self.mP * d2a ;
        d2norm_Tp = (np.dot(dTp, dTp) + np.dot(Tp, d2Tp) - dnorm_Tp**2) / norm_Tp
        d2q = (d2Tp - dq * dnorm_Tp - q * d2norm_Tp - dq * dnorm_Tp) / norm_Tp

        return x, v, a, da, q, dq, d2q


class SmoothTraj5Payload(Trajectory):
    def __init__(self, x0, xf, tf=10):
        super().__init__(tf)
        self._x0 = np.array(x0)
        self._xf = np.array(xf)
        self.compute_traj_params()

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0, 0, 1])

    def compute_traj_params(self):
        a = self._xf - self._x0
        self._pos_params = np.array([self._x0, np.zeros(3), np.zeros(3), 10 * a, -15 * a, 6 * a])
        self._vel_params = np.array([np.zeros(3), 2 * np.zeros(3), 3 * 10 * a, 4 * (-15) * a, 5 * 6 * a, np.zeros(3)])
        self._acc_params = np.array([np.zeros(3), 6 * 10 * a, 12 * (-15) * a, 20 * 6 * a, np.zeros(3), np.zeros(3)])

    def get(self, t):
        tau = t / self._tf
        tau_vec = np.array([1, tau, tau**2, tau**3, tau**4, tau**5]).reshape(-1, 1)

        x = np.dot(self._pos_params.T, tau_vec).flatten()
        v = np.dot(self._vel_params.T, tau_vec).flatten()
        a = np.dot(self._acc_params.T, tau_vec).flatten()
        
        # Compute higher-order derivatives for payload
        da = np.gradient(a, t)
        d2a = np.gradient(da, t)

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
        self.e3 = np.array([0,0,1])

    def get(self, t):
        w1 = 2 * np.pi * self.f1
        w2 = 2 * np.pi * self.f2
        w3 = 2 * np.pi * self.f3

        x = np.array([
            self.ax * (np.sin(w1 * t + self.phix)),
            self.ay * (1 - np.cos(w2 * t + self.phiy)),
            self.az * (-1 - np.cos(w3 * t + self.phiz))
        ])
        v = np.array([
            self.ax * np.cos(w1 * t + self.phix) * w1,
            self.ay * np.sin(w2 * t + self.phiy) * w2,
            self.az * np.sin(w3 * t + self.phiz) * w3
        ])
        a = np.array([
            -self.ax * np.sin(w1 * t + self.phix) * w1 * w1,
            self.ay * np.cos(w2 * t + self.phiy) * w2 * w2,
            self.az * np.cos(w3 * t + self.phiz) * w3 * w3
        ])
        da = np.array([
            -self.ax * np.cos(w1 * t + self.phix) * w1 * w1 * w1,
            -self.ay * np.sin(w2 * t + self.phiy) * w2 * w2 * w2,
            -self.az * np.sin(w3 * t + self.phiz) * w3 * w3 * w3
        ])
        d2a = np.array([
            self.ax * np.sin(w1 * t + self.phix) * w1 * w1 * w1 * w1,
            -self.ay * np.cos(w2 * t + self.phiy) * w2 * w2 * w2 * w2,
            -self.az * np.cos(w3 * t + self.phiz) * w3 * w3 * w3 * w3
        ])
        
        Tp = -self.mP * (a + self.g * self.e3)
        norm_Tp = np.linalg.norm(Tp)
        q = Tp / norm_Tp

        dTp = -self.mP * da
        dnorm_Tp = 1 / norm_Tp * np.dot(Tp, dTp)
        dq = (dTp - q * dnorm_Tp) / norm_Tp

        d2Tp = -self.mP * d2a ;
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
        return (L * k * exp_term) / (1 + exp_term)**2

    def sigmoid_second_derivative(self, t, L, k, t0):
        exp_term = np.exp(-k * (t - t0))
        return L * k**2 * exp_term * (1 + exp_term)**(-2) * (-1 + 2 * exp_term / (1 + exp_term))

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
            x = np.array([
                self.sigmoid(t, self.L_x, self.k_x, self.t0_x),
                0,
                self.double_sigmoid_decrease(t, self.L_z, self.k_z, self.t0_z, self.d_z)
            ])
            v = np.array([
                self.sigmoid_derivative(t, self.L_x, self.k_x, self.t0_x),
                0,
                self.double_sigmoid_decrease_derivative(t, self.L_z, self.k_z, self.t0_z, self.d_z)
            ])
            a = np.array([
                self.sigmoid_second_derivative(t, self.L_x, self.k_x, self.t0_x),
                0,
                self.double_sigmoid_decrease_second_derivative(t, self.L_z, self.k_z, self.t0_z, self.d_z)
            ])
        else:
            dt = t - self._tf
            x_tf = np.array([
                self.sigmoid(self._tf, self.L_x, self.k_x, self.t0_x),
                0,
                self.double_sigmoid_decrease(self._tf, self.L_z, self.k_z, self.t0_z, self.d_z)
            ])
            v_tf = np.array([
                self.sigmoid_derivative(self._tf, self.L_x, self.k_x, self.t0_x),
                0,
                self.double_sigmoid_decrease_derivative(self._tf, self.L_z, self.k_z, self.t0_z, self.d_z)
            ])
            a_tf = np.array([
                self.sigmoid_second_derivative(self._tf, self.L_x, self.k_x, self.t0_x),
                0,
                self.double_sigmoid_decrease_second_derivative(self._tf, self.L_z, self.k_z, self.t0_z, self.d_z)
            ])
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
        self.phix = np.random.choice([np.pi/2, 3*np.pi/2])
        self.phiy = np.random.choice([np.pi/2, 3*np.pi/2])
        self.phiz = np.random.choice([np.pi/2, 3*np.pi/2])

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0,0,1])

    def get(self, t):
        w1 = 2 * np.pi * self.f1
        w2 = 2 * np.pi * self.f2
        w3 = 2 * np.pi * self.f3

        x = np.array([
            self.ax * (1 - np.cos(w1 * t + self.phix)),
            self.ay * (1 - np.cos(w2 * t + self.phiy)),
            self.az * (1 - np.cos(w3 * t + self.phiz))
        ])
        v = np.array([
            self.ax * np.sin(w1 * t + self.phix) * w1,
            self.ay * np.sin(w2 * t + self.phiy) * w2,
            self.az * np.sin(w3 * t + self.phiz) * w3
        ])
        a = np.array([
            self.ax * np.cos(w1 * t + self.phix) * w1 * w1,
            self.ay * np.cos(w2 * t + self.phiy) * w2 * w2,
            self.az * np.cos(w3 * t + self.phiz) * w3 * w3
        ])
        da = np.array([
            -self.ax * np.sin(w1 * t + self.phix) * w1 * w1 * w1,
            -self.ay * np.sin(w2 * t + self.phiy) * w2 * w2 * w2,
            -self.az * np.sin(w3 * t + self.phiz) * w3 * w3 * w3
        ])
        d2a = np.array([
            -self.ax * np.cos(w1 * t + self.phix) * w1 * w1 * w1 * w1,
            -self.ay * np.cos(w2 * t + self.phiy) * w2 * w2 * w2 * w2,
            -self.az * np.cos(w3 * t + self.phiz) * w3 * w3 * w3 * w3
        ])
        
        Tp = -self.mP * (a + self.g * self.e3)
        norm_Tp = np.linalg.norm(Tp)
        q = Tp / norm_Tp

        dTp = -self.mP * da
        dnorm_Tp = 1 / norm_Tp * np.dot(Tp, dTp)
        dq = (dTp - q * dnorm_Tp) / norm_Tp

        d2Tp = -self.mP * d2a ;
        d2norm_Tp = (np.dot(dTp, dTp) + np.dot(Tp, d2Tp) - dnorm_Tp**2) / norm_Tp
        d2q = (d2Tp - dq * dnorm_Tp - q * d2norm_Tp - dq * dnorm_Tp) / norm_Tp

        q0 = rodriguesExpm(np.array([0,1,0]), np.pi/6) @ q
        q1 = rodriguesExpm(np.array([0,1,0]), -np.pi/6) @ q

        return x, v, a, da, q0, q1, q, dq, d2q
    
    def plot3d_payload_multiple(self):
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
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label='Trajectory', color='b', linewidth=2)
        ax.set_title('3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        max_range = np.array([x[:, 0].max()-x[:, 0].min(), x[:, 1].max()-x[:, 1].min(), x[:, 2].max()-x[:, 2].min()]).max() / 2.0
        mid_x = (x[:, 0].max()+x[:, 0].min()) * 0.5
        mid_y = (x[:, 1].max()+x[:, 1].min()) * 0.5
        mid_z = (x[:, 2].max()+x[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        step = len(T) // 20
        for i in range(0, len(T), step):
            ax.quiver(x[i, 0], x[i, 1], x[i, 2], -q0[i, 0], -q0[i, 1], -q0[i, 2], length=0.5, normalize=True, color='g',  arrow_length_ratio=0.01)
            ax.quiver(x[i, 0], x[i, 1], x[i, 2], -q1[i, 0], -q1[i, 1], -q1[i, 2], length=0.5, normalize=True, color='b',  arrow_length_ratio=0.01)
            ax.quiver(x[i, 0], x[i, 1], x[i, 2], -q[i, 0], -q[i, 1], -q[i, 2], length=0.5, normalize=True, color='r',  arrow_length_ratio=0.01)

        plt.show()


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

        self.phix = np.random.choice([np.pi/2, 3*np.pi/2])
        self.phiy = np.random.choice([np.pi/2, 3*np.pi/2])
        self.phiz = np.random.choice([np.pi/2, 3*np.pi/2])

        self.mP = 0.1
        self.g = 9.8
        self.e3 = np.array([0,0,1])

    def get(self, t):
        w1 = 2 * np.pi * self.f1
        w2 = 2 * np.pi * self.f2
        w3 = 2 * np.pi * self.f3

        x = np.array([
            self.ax * (1 - np.cos(w1 * t + self.phix)),
            self.ay * (1 - np.cos(w2 * t + self.phiy)),
            self.az * (1 - np.cos(w3 * t + self.phiz))
        ])
        v = np.array([
            self.ax * np.sin(w1 * t + self.phix) * w1,
            self.ay * np.sin(w2 * t + self.phiy) * w2,
            self.az * np.sin(w3 * t + self.phiz) * w3
        ])
        a = np.array([
            self.ax * np.cos(w1 * t + self.phix) * w1 ** 2,
            self.ay * np.cos(w2 * t + self.phiy) * w2 ** 2,
            self.az * np.cos(w3 * t + self.phiz) * w3 ** 2
        ])
        da = np.array([
            -self.ax * np.sin(w1 * t + self.phix) * w1 ** 3,
            -self.ay * np.sin(w2 * t + self.phiy) * w2 ** 3,
            -self.az * np.sin(w3 * t + self.phiz) * w3 ** 3
        ])
        d2a = np.array([
            -self.ax * np.cos(w1 * t + self.phix) * w1 ** 4,
            -self.ay * np.cos(w2 * t + self.phiy) * w2 ** 4,
            -self.az * np.cos(w3 * t + self.phiz) * w3 ** 4
        ])
        d3a = np.array([
            self.ax * np.sin(w1 * t + self.phix) * w1 ** 5,
            self.ay * np.sin(w2 * t + self.phiy) * w2 ** 5,
            self.az * np.sin(w3 * t + self.phiz) * w3 ** 5
        ])
        d4a = np.array([
            -self.ax * np.cos(w1 * t + self.phix) * w1 ** 6,
            -self.ay * np.cos(w2 * t + self.phiy) * w2 ** 6,
            -self.az * np.cos(w3 * t + self.phiz) * w3 ** 6
        ])
        
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
        d3norm_Tp = (2 * np.dot(d2Tp, dTp) + np.dot(dTp, d2Tp) + np.dot(Tp, d3Tp) - 3 * dnorm_Tp * d2norm_Tp) / norm_Tp
        d3q = (d3Tp - d2q * dnorm_Tp - dq * d2norm_Tp - dq * d2norm_Tp - q * d3norm_Tp - d2q * dnorm_Tp - dq * d2norm_Tp - d2q * dnorm_Tp) / norm_Tp

        d4Tp = -self.mP * d4a
        d4norm_Tp = (2 * np.dot(d3Tp, dTp)+2*np.dot(d2Tp, d2Tp) + np.dot(d2Tp, d2Tp)+np.dot(dTp, d3Tp) + np.dot(dTp, d3Tp)+np.dot(Tp, d4Tp) - 3*d2norm_Tp**2-3*dnorm_Tp*d3norm_Tp - d3norm_Tp*dnorm_Tp) / norm_Tp
        d4q = (d4Tp - d3q*dnorm_Tp-d2q*d2norm_Tp - d2q*d2norm_Tp-dq*d3norm_Tp - d2q*d2norm_Tp-dq*d3norm_Tp - dq*d3norm_Tp-q*d4norm_Tp - d3q*dnorm_Tp-d2q*d2norm_Tp - d2q*d2norm_Tp-dq*d3norm_Tp - d3q*dnorm_Tp-d2q*d2norm_Tp - d3q*dnorm_Tp ) / norm_Tp

        return x, v, a, da, d2a, d3a, d4a, q, dq, d2q, d3q, d4q


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


""" TEST """
# region
# class FullCrazyTrajectory(Trajectory):
#     def __init__(self, traj, tf=45):
#         """
#         Full trajectory that includes:
#         1. Smooth takeoff (0-5s) using a QuinticTrajectory.
#         2. Crazy trajectory (5-35s).
#         3. Smooth landing (35-45s) using a QuinticTrajectory.

#         Args:
#             traj (Trajectory): An instance of CrazyTrajectory.
#             tf (float): Total duration of the trajectory (default: 45s).
#         """
#         super().__init__(tf)
#         self.crazy_traj = traj
#         self.takeoff_height = 1.5  # Target height for takeoff and landing

#         # Define takeoff and landing trajectories using QuinticTrajectory
#         self.takeoff_traj = QuinticTrajectory(tf=5, x0=np.array([0, 0, 0]), xf=np.array([0, 0, self.takeoff_height]))
#         self.landing_traj = QuinticTrajectory(tf=10, x0=np.array([0, 0, self.takeoff_height]), xf=np.array([0, 0, 0]))

#     def get(self, t):
#         if t < 5:
#             return self.takeoff_traj.get(t)  # Takeoff phase
#         elif t < 35:
#             crazy_x, crazy_v, crazy_a = self.crazy_traj.get(t - 5)
#             crazy_x[2] += self.takeoff_height  # Shift trajectory to hover at 1.5m height
#             return crazy_x, crazy_v, crazy_a
#         else:
#             return self.landing_traj.get(t - 35)  # Landing phase
# endregion


""" TRAIN """
# region
class FullCrazyTrajectory(Trajectory):
    def __init__(self,
                 traj=CrazyTrajectory(tf=30, ax=0, ay=0, az=0, f1=0, f2=0, f3=0),
                 tf=45):
        super().__init__(tf)
        self.takeoff_traj = SmoothTraj5(x0=np.array([0, 0, 0]), xf=np.array([0, 0, 1.5]), tf=5)
        self.crazy_traj = traj
        self.landing_traj = None
        self.takeoff_time = 5
        self.landing_time = 35

    def get(self, t):
        if t < self.takeoff_time:  # Takeoff Phase
            return self.takeoff_traj.get(t)
        elif t < self.landing_time:  # Crazy Trajectory Phase
            x, v, a = self.crazy_traj.get(t - self.takeoff_time)
            x += np.array([0, 0, 1.5])
            return x, v, a
        else:  # Landing Phase
            if self.landing_traj is None:
                final_pos, _, _ = self.crazy_traj.get(self.landing_time - self.takeoff_time)
                final_pos += np.array([0, 0, 1.5])
                self.landing_traj = SmoothTraj5(x0=final_pos, xf=[final_pos[0], final_pos[1], 0], tf=self._tf-self.landing_time)
            return self.landing_traj.get(t - self.landing_time)
# endregion

if __name__ == "__main__":
    traj = FullCrazyTrajectory(tf=45, traj=CrazyTrajectory(tf=30, ax=1, ay=1, az=1, f1=0.5, f2=0.5, f3=0.5))
    # traj = FullCrazyTrajectory(tf=45, traj=CrazyTrajectory(tf=30, ax=0, ay=0, az=0, f1=0, f2=0, f3=0))
    # traj.plot()
    # traj.plot3d()

    # smooth_traj = SmoothTraj5(x0=np.array([0,0,0]), xf=np.array([0,0,1]), tf=5)
    # smooth_traj.plot()

    # quintic_traj = QuinticTrajectory(x0=np.array([0,0,0]), xf=np.array([0,0,1]), tf=5)
    # quintic_traj.plot()
