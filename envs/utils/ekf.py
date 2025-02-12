import numpy as np
from scipy.linalg import expm
from geo_tools import hat
from rotation_transformations import *

class EKFQuadrotor:
    def __init__(self, dt, mQ, JQ):
        self.dt = dt  # Sampling time
        self.s_dim = 18  # [x (3), R (9), v (3), ω (3)]

        # Initial state vector
        self.s = np.zeros((self.s_dim, 1))

        # Initial covariance matrix
        self.P = np.eye(self.s_dim) * 0.1

        # Process and measurement noise covariance matrices
        self.Q = np.eye(self.s_dim) * 0.01  # Process noise
        self.R = np.eye(self.s_dim) * 0.05  # Measurement noise

        # Quadrotor property
        self.mQ = mQ
        self.JQ = JQ
        self.mQ_inv = 1 / mQ
        self.JQ_inv = np.linalg.inv(JQ)
        self.l = 0.1524
        self.d = self.l / np.sqrt(2)
        self.κ = 0.025
        self.srt2ctbr = np.array([[1, 1, 1, 1],
                                  [self.d, -self.d, -self.d, self.d],
                                  [-self.d, -self.d, self.d, self.d],
                                  [-self.κ, self.κ, -self.κ, self.κ]])
        
        # World property
        self.g = 9.81
        self.e3 = np.array([0, 0, 1])

    def f(self, s, a):
        """Quadrotor dynamics model"""
        x = s[:3]
        R = s[3:12].reshape(3, 3)
        v = s[12:15]
        ω = s[15:]

        ctbr = self.srt2ctbr @ a
        f = ctbr[0]
        M = ctbr[1:]

        x_new = x + v * self.dt
        R_new = R @ expm(hat(ω) * self.dt)
        v_new = v + (-self.g * self.e3 + self.mQ_inv * f * R @ self.e3) * self.dt
        ω_new = ω + self.JQ_inv @ (-np.cross(ω, self.JQ @ ω) + M) * self.dt

        return np.hstack([x_new, R_new.flatten(), v_new, ω_new])

    def jacobian_f(self, x, u):
        """Compute the Jacobian matrix F of the state transition function"""
        F = np.eye(self.s_dim)
        F[:3, 12:15] = np.eye(3) * self.dt  # Partial derivative dx/dv
        return F

    def predict(self, a):
        """Prediction step of the EKF"""
        self.s = self.f(self.s, a)  # Predict next state using dynamics model
        F = self.jacobian_f(self.s, a)  # Compute Jacobian matrix
        self.P = F @ self.P @ F.T + self.Q  # Update covariance matrix

    def update(self, z):
        """Update step using sensor measurements"""
        K = self.P @ np.linalg.inv(self.P + self.R)  # Compute Kalman gain
        self.s = self.s + K @ (z - self.s)  # Update state estimate
        self.P = (np.eye(self.s_dim) - K) @ self.P  # Update covariance matrix


if __name__ == "__main__":
    JQ = np.array([[0.49, 0, 0],
                   [0, 0.53, 0],
                   [0, 0, 0.98]]) * 1e-2
    
    ekf = EKFQuadrotor(dt=0.002, mQ=0.8, JQ=JQ)
    
    x = np.zeros(3)
    R = np.eye(3)
    v = np.array([0.5, 0.4, 0.3])
    ω = np.array([0.1, 0.2, 0.3])

    s = np.concatenate([x, R.flatten(), v, ω])
    s_next = ekf.f(s=s, a=[4, 3, 3, 4])

    print(np.round(s_next, 4))
    print(rot2euler(s_next.flatten()[3:12].reshape(3, 3)))