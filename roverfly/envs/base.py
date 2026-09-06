"""Shared mechanics for the single-quadrotor environments."""

import numpy as np
from gymnasium.spaces import Box

from roverfly.simulation.environment import MujocoEnv
from roverfly.simulation.viewer import setup_viewer


class BaseQuadrotorEnv(MujocoEnv):
    """Common delay, disturbance, rotor, and rendering behavior."""

    def _set_observation_space(self):
        size = self.o_dim if self.is_io_history else self.s_dim
        return Box(low=-np.inf, high=np.inf, shape=(size,))

    def _action_delay(self):
        if self.data.time - self.action_queue[0][0] >= self.delay_time:
            action = self.action_queue.popleft()[1]
            self.delay_time = np.random.uniform(*self.delay_range)
            return action
        return self.action_queue[0][1]

    def _rotor_dynamics(self, desired_forces):
        time_constants = np.where(
            desired_forces > self.actual_forces,
            self.tau_up,
            self.tau_down,
        )
        alpha = self.sim_dt / (time_constants + self.sim_dt)
        return (1 - alpha) * self.actual_forces + alpha * desired_forces

    def _apply_downwash(self):
        if self.data.qpos[2] >= 0.5:
            return
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi / 2)
        direction = np.array(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ]
        )
        self.data.xfrc_applied[self.quadrotor_body_id, :3] = direction * (0.5 - self.data.qpos[2])

    def _apply_disturbance(self):
        elapsed = self.data.time - self.disturbance_start
        if elapsed < self.disturbance_duration:
            pass
        elif elapsed < self.disturbance_duration + 1:
            self.disturbance_wrench = np.zeros(6)
        else:
            self.disturbance_duration = np.random.uniform(*self.disturbance_duration_range)
            force = np.random.uniform(*self.force_disturbance_range, size=3)
            torque = np.random.uniform(*self.torque_disturbance_range, size=3)
            self.disturbance_wrench = np.concatenate([force, torque])
            self.disturbance_start = self.data.time
        self.data.xfrc_applied[self.quadrotor_body_id, :6] = self.disturbance_wrench

    def _truncated(self):
        return False

    def render(self):
        if self.mujoco_renderer.viewer is not None and self.timestep <= 1:
            setup_viewer(self.mujoco_renderer.viewer)
        return self.mujoco_renderer.render(self.render_mode)
