"""
    Description:
        Environment for multiple quadrotor payload trajectory tracking
        without time delay
    Objective:
        Quadrotor trajectory tracking
"""

# Helpers
import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
import warnings
# Mujoco
import mujoco as mj
from mujoco_gym.mujoco_env import MujocoEnv
from envs.quadrotor_random_env import QuadrotorRandomEnv
# Gym
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import utils
# ETC
import envs.utils.utility_trajectory as ut
from envs.utils.rotation_transformations import *
from envs.utils.render_util import setup_viewer
from envs.ppo.ppo import PPO # Customized
# SB3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv, is_vecenv_wrapped
from stable_baselines3.common import type_aliases
from stable_baselines3.common.evaluation import evaluate_policy
import argparse


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 10.0}
  
class QuadrotorMultipleEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    def __init__(
        self,
        max_timesteps = 10000,
        xml_file: str = "../assets/quadrotor_x_cfg_payload_multiple.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 1.0,
        env_num: int = 0,
        **kwargs
    ):
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)
        self.frame_skip = frame_skip
        # self.render_mode = "human"

        self.num_quadrotors: int = 3
        # self.env_Q1 = QuadrotorRandomEnv()
        self.random_envs: List[QuadrotorRandomEnv] = [QuadrotorRandomEnv() for _ in range(self.num_quadrotors)]

        ##################################################
        #################### DYNAMICS ####################
        ##################################################
        # region
        self.sim_freq: float       = 500.0
        self.policy_freq: float    = 500.0
        self.sim_dt: float         = 1 / self.sim_freq
        self.policy_dt: float      = 1 / self.policy_freq
        self.num_sims_per_env_step = int(self.sim_freq // self.policy_freq)
        # endregion    
        ##################################################
        ###################### TIME ######################
        ##################################################
        # region
        self.max_timesteps: int = max_timesteps
        self.timestep: int      = 0
        self.time_in_sec: float = 0.0
        self.time = 0
        # endregion
        ##################################################
        ################## OBSERVATION ###################
        ##################################################
        # region
        # Lengths
        self.n_action: int      = 4 * self.num_quadrotors
        self.n_observation: int = 108 * self.num_quadrotors
        self.num_episode: int   = 0
        self.history_len        = 5
        self.future_len         = 3
        self.n_delay            = 0
        # Buffers
        self.e_buffer_Q1        = deque(np.zeros((self.history_len, 6)), maxlen=self.history_len)
        self.e_buffer_Q2        = deque(np.zeros((self.history_len, 6)), maxlen=self.history_len)
        self.e_buffer_Q3        = deque(np.zeros((self.history_len, 6)), maxlen=self.history_len)
        self.a_buffer_Q1        = deque(np.zeros((self.history_len, 4)), maxlen=self.history_len)
        self.a_buffer_Q2        = deque(np.zeros((self.history_len, 4)), maxlen=self.history_len)
        self.a_buffer_Q3        = deque(np.zeros((self.history_len, 4)), maxlen=self.history_len)
        # Episode
        self.history_epi        = {'setpoint': deque([0]*10, maxlen=10),
                                   'curve': deque([0]*10, maxlen=10)}
        self.progress           = {'setpoint': 1e-3,
                                   'curve': 1e-3}
        # Spaces
        self.action_space       = self._set_action_space()
        self.observation_space  = self._set_observation_space()
        # endregion
        ##################################################
        ################### MUJOCOENV ####################
        ##################################################
        # region
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=self.observation_space, default_camera_config=default_camera_config, **kwargs)
        # HACK
        self.action_space = self._set_action_space()
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.sim_dt))
        }
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }
        self._reset_noise_scale = reset_noise_scale
        # endregion
        ##################################################
        ################### PD CONTROL ###################
        ##################################################
        # region
        self.mQ = 0.8
        self.mP = 0.1
        self.g = 9.81
        self.JQ = np.array([[0.49, 0, 0],
                            [0, 0.53, 0],
                            [0, 0, 0.98]]) * 1e-2
        self.kPdφ, self.kPdθ, self.kPdψ = 1.0, 1.0, 1.0  # 1.0, 1.0, 1.0  # 0.5, 0.5, 0.5  
        self.kIdφ, self.kIdθ, self.kIdψ = 0.2, 0.2, 0.2  # 0.2, 0.2, 0.2  # 0.1, 0.1, 0.1  
        self.kDdφ, self.kDdθ, self.kDdψ = 0.002, 0.002, 0.002  # 0.002, 0.002, 0.002  # 0.001, 0.001, 0.001 
        
        self.clipI = 0.3

        self.edφI, self.edθI, self.edψI = 0, 0, 0
        self.edφP_prev, self.edθP_prev, self.edψP_prev = 0, 0, 0

        self.l = 0.1524
        self.d = self.l / np.sqrt(2)
        self.κ = 0.025
        self.A = np.linalg.inv(np.array([[1, 1, 1, 1],
                                         [self.d, -self.d, -self.d, self.d],
                                         [-self.d, -self.d, self.d, self.d],
                                         [-self.κ, self.κ, -self.κ, self.κ]]))
        self.CR, self.wb, self.cT = 1148, -141.4, 1.105e-5
        # endregion

    def _set_action_space(self):
        low = np.array([-1, -1, -1, -1] * 3)
        high = np.array([1, 1, 1, 1] * 3)
        self.action_space = Box(low=low, high=high)
        return self.action_space

    def _set_observation_space(self):
        obs_shape = self.n_observation
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        return observation_space

    def _init_history_ff(self):
        [self.e_buffer_Q1.append(np.copy(self.e_curr_Q1)) for _ in range(self.history_len)]
        [self.e_buffer_Q2.append(np.copy(self.e_curr_Q2)) for _ in range(self.history_len)]
        [self.e_buffer_Q3.append(np.copy(self.e_curr_Q3)) for _ in range(self.history_len)]
        [self.a_buffer_Q1.append(np.copy(self.action[0:4])) for _ in range(self.history_len)]
        [self.a_buffer_Q2.append(np.copy(self.action[4:8])) for _ in range(self.history_len)]
        [self.a_buffer_Q3.append(np.copy(self.action[8:12])) for _ in range(self.history_len)]

    def reset(self, seed=None, randomize=None):
        self.action = np.zeros(self.n_action)
        self.e_curr_Q1 = np.zeros(6)
        self.e_curr_Q2 = np.zeros(6)
        self.e_curr_Q3 = np.zeros(6)
        self._init_history_ff()

        self._reset_env()
        self._reset_model()
        if self.render_mode == 'human': self._reset_renderer()

        obs = self._get_obs()
        self.info = self._get_reset_info()

        return obs, self.info

    def _reset_env(self):
        self.timestep     = 0  # discrete timestep, k
        self.time_in_sec  = 0.0  # time
        self.total_reward = 0
        self.terminated   = None
        self.info         = {}

    def _reset_model(self):
        # self.progress['setpoint'] = 0
        # self.progress['curve'] = 0.2

        """ Choose trajectory type """
        if self.progress['setpoint'] < 0.5:
            self.stage = 1
            self.traj_type = np.random.choice(['setpoint', 'curve'], p=[0.9, 0.1])
        else:
            self.stage = 2
            self.traj_type = np.random.choice(['setpoint', 'curve'], p=[0.1, 0.9])
        
        self.traj_type = 'curve'

        """ Relative positions """
        rot = euler2rot(np.array([0, 0, np.random.uniform(0, 2*np.pi)]))
        self.xPr = rot @ np.array([0.5, -0.289, -0.866])
        self.xQ2r = rot @ np.array([1, 0, 0])
        self.xQ3r = rot @ np.array([0.5, -0.866, 0])

        """ Generate trajectories """
        self._generate_trajectories()

        """ Initial Position """
        xQ10 = self.xQd[0, 0]
        xP0 = xQ10 + self.xPr
        self.xQ20 = xQ10 + self.xQ2r
        self.xQ30 = xQ10 + self.xQ3r

        vQ10 = self.vQd[0, 0]
        vP0 = self.vQd[0, 0]
        vQ20 = self.vQd[0, 0]
        vQ30 = self.vQd[0, 0]
        
        qpos = np.concatenate([xQ10, self.init_qpos[3:7], self.init_qpos[7:11], self.xQ20, self.init_qpos[14:18], self.init_qpos[18:22], self.xQ30, self.init_qpos[25:33], xP0, self.init_qpos[36:44]])
        qvel = np.concatenate([vQ10, self.init_qvel[3:6], self.init_qvel[6:9], vQ20, self.init_qvel[12:15], self.init_qvel[15:18], vQ30, self.init_qvel[21:27], vP0, self.init_qvel[30:36]])
        self.set_state(qpos, qvel)

    def _generate_trajectories(self):
        """ Set trajectory parameters """
        if self.traj_type == 'setpoint':
            self.traj = ut.CrazyTrajectory(tf=self.max_timesteps*self.policy_dt, ax=0, ay=0, az=0, f1=0, f2=0, f3=0)
            self.difficulty = self.stage * self.progress["setpoint"]
        if self.traj_type == 'curve':
            self.traj = ut.CrazyTrajectory(tf=self.max_timesteps*self.policy_dt,
                                           ax=np.random.choice([-1,1])*3*self.progress["curve"],
                                           ay=np.random.choice([-1,1])*0*self.progress["curve"],
                                           az=np.random.choice([-1,1])*0*self.progress["curve"],
                                           f1=np.random.choice([-1,1])*0.5*self.progress["curve"],
                                           f2=np.random.choice([-1,1])*0.5*self.progress["curve"],
                                           f3=np.random.choice([-1,1])*0.5*self.progress["curve"],
                                          )
            self.difficulty = self.stage * self.progress["curve"]

        """ Compute trajectory """
        self.xQd = np.zeros((self.num_quadrotors, self.max_timesteps + self.history_len + self.n_delay, 3))
        self.vQd = np.zeros((self.num_quadrotors, self.max_timesteps + self.history_len + self.n_delay, 3))
      
        for i in range(self.n_delay):
            self.xQd[0, i], self.vQd[0, i], _ = self.traj.get(0)
            self.xQd[1, i], self.vQd[1, i] = self.xQd[0, i] + self.xQ2r, self.vQd[0, i]
            self.xQd[2, i], self.vQd[2, i] = self.xQd[0, i] + self.xQ3r, self.vQd[0, i]
        for i in range(self.n_delay, self.max_timesteps + self.history_len + self.n_delay):
            self.xQd[0, i], self.vQd[0, i], _ = self.traj.get((i - self.n_delay) * self.policy_dt)
            self.xQd[1, i], self.vQd[1, i] = self.xQd[0, i] + self.xQ2r, self.vQd[0, i]
            self.xQd[2, i], self.vQd[2, i] = self.xQd[0, i] + self.xQ3r, self.vQd[0, i]

    def _reset_renderer(self):
        self.render()
        setup_viewer(self.mujoco_renderer.viewer)
        del self.mujoco_renderer.viewer._markers[:]

    def _get_obs(self):
        # Present
        self._get_obs_curr()  # Set s_Q1, s_Q2, s_Q3

        # Past
        e_buffer_Q1 = np.array(self.e_buffer_Q1).flatten()  # 30
        e_buffer_Q2 = np.array(self.e_buffer_Q2).flatten()  # 30
        e_buffer_Q3 = np.array(self.e_buffer_Q3).flatten()  # 30

        a_buffer_Q1 = np.array(self.a_buffer_Q1).flatten()  # 20
        a_buffer_Q2 = np.array(self.a_buffer_Q2).flatten()  # 20
        a_buffer_Q3 = np.array(self.a_buffer_Q3).flatten()  # 20

        io_history_Q1 = np.concatenate([e_buffer_Q1, a_buffer_Q1])  # 50
        io_history_Q2 = np.concatenate([e_buffer_Q2, a_buffer_Q2])  # 50
        io_history_Q3 = np.concatenate([e_buffer_Q3, a_buffer_Q3])  # 50

        # Future
        xQ1_ff = self.xQ1 - self.xQd[0, self.timestep : self.timestep + self.future_len]
        xQ2_ff = self.xQ2 - self.xQd[1, self.timestep : self.timestep + self.future_len]
        xQ3_ff = self.xQ3 - self.xQd[2, self.timestep : self.timestep + self.future_len]
        
        vQ1_ff = self.vQ1 - self.vQd[0, self.timestep : self.timestep + self.future_len]
        vQ2_ff = self.vQ2 - self.vQd[1, self.timestep : self.timestep + self.future_len]
        vQ3_ff = self.vQ3 - self.vQd[2, self.timestep : self.timestep + self.future_len]
        
        ff_Q1 = np.concatenate([(xQ1_ff @ self.RQ1).flatten(), (vQ1_ff @ self.RQ1).flatten()])  # 18
        ff_Q2 = np.concatenate([(xQ2_ff @ self.RQ2).flatten(), (vQ2_ff @ self.RQ2).flatten()])  # 18
        ff_Q3 = np.concatenate([(xQ3_ff @ self.RQ3).flatten(), (vQ3_ff @ self.RQ3).flatten()])  # 18

        obs_full_Q1 = np.concatenate([self.obs_curr_Q1, io_history_Q1, ff_Q1])  # 40+50+18 = 108
        obs_full_Q2 = np.concatenate([self.obs_curr_Q2, io_history_Q2, ff_Q2])  # 40+50+18 = 108
        obs_full_Q3 = np.concatenate([self.obs_curr_Q3, io_history_Q3, ff_Q3])  # 40+50+18 = 108

        obs_full = np.concatenate([obs_full_Q1, obs_full_Q2, obs_full_Q3])

        return obs_full

    def _get_obs_curr(self):
        # 1. Get all info needed
        self.xQ1 = self.data.qpos[0:3]
        self.xQ2 = self.data.qpos[11:14]
        self.xQ3 = self.data.qpos[22:25]
        self.xP = self.data.qpos[33:36]

        self.vQ1 = self.data.qvel[0:3]
        self.vQ2 = self.data.qvel[9:12]
        self.vQ3 = self.data.qvel[18:21]
        self.vP = self.data.qvel[27:30]

        self.RQ1 = quat2rot(self.data.qpos[3:7])
        self.RQ2 = quat2rot(self.data.qpos[14:18])
        self.RQ3 = quat2rot(self.data.qpos[25:29])

        self.ωQ1 = self.data.qvel[3:6]
        self.ωQ2 = self.data.qvel[12:15]
        self.ωQ3 = self.data.qvel[21:24]

        # 2. Re-organize and make random env obs
        self.exQ1 = self.xQ1 - self.xQd[0, self.timestep]
        self.exQ2 = self.xQ2 - self.xQd[1, self.timestep]
        self.exQ3 = self.xQ3 - self.xQd[2, self.timestep]

        self.evQ1 = self.vQ1 - self.vQd[0, self.timestep]
        self.evQ2 = self.vQ2 - self.vQd[1, self.timestep]
        self.evQ3 = self.vQ3 - self.vQd[2, self.timestep]

        self.e_curr_Q1 = np.concatenate([self.RQ1.T @ self.exQ1, self.RQ1.T @ self.evQ1])
        self.e_curr_Q2 = np.concatenate([self.RQ2.T @ self.exQ2, self.RQ2.T @ self.evQ2])
        self.e_curr_Q3 = np.concatenate([self.RQ3.T @ self.exQ3, self.RQ3.T @ self.evQ3])

        self.obs_curr_Q1 = np.concatenate([self.e_curr_Q1,
                                     self.RQ1.T @ (self.xQ1 - self.xP), self.RQ1.T @ (self.vQ1 - self.vP), 
                                     self.RQ1.T @ (self.xQ1 - self.xQ2), self.RQ1.T @ (self.vQ1 - self.vQ2),
                                     self.RQ1.T @ (self.xQ1 - self.xQ3), self.RQ1.T @ (self.vQ1 - self.vQ3),
                                     self.action[0:4], self.RQ1.flatten(), self.ωQ1])  # 40
        
        self.obs_curr_Q2 = np.concatenate([self.e_curr_Q2,
                                     self.RQ2.T @ (self.xQ2 - self.xP), self.RQ2.T @ (self.vQ2 - self.vP), 
                                     self.RQ2.T @ (self.xQ2 - self.xQ3), self.RQ2.T @ (self.vQ2 - self.vQ3),
                                     self.RQ2.T @ (self.xQ2 - self.xQ1), self.RQ2.T @ (self.vQ2 - self.vQ1),
                                     self.action[4:8], self.RQ2.flatten(), self.ωQ2])  # 40
        
        self.obs_curr_Q3 = np.concatenate([self.e_curr_Q3,
                                     self.RQ3.T @ (self.xQ3 - self.xP), self.RQ3.T @ (self.vQ3 - self.vP), 
                                     self.RQ3.T @ (self.xQ3 - self.xQ1), self.RQ3.T @ (self.vQ3 - self.vQ1),
                                     self.RQ3.T @ (self.xQ3 - self.xQ2), self.RQ3.T @ (self.vQ3 - self.vQ2),
                                     self.action[4:8], self.RQ3.flatten(), self.ωQ3])  # 40

    def step(self, action, restore=False):
        self.action = action
        for _ in range(self.num_sims_per_env_step):
            self.do_simulation(self.action, self.frame_skip)  # a_{t}
        obs_full = self._get_obs()  # [o^{Q1}_{t}, o^{Q2}_{t}, o^{Q3}_{t}]
        if self.render_mode == 'human':
            self.render()
        reward, reward_dict = self._get_reward()
        self.info["reward_dict"] = reward_dict
        terminated = self._terminated()
        truncated = self._truncated()
        self._update_data(reward=reward, step=True)

        return obs_full, reward, terminated, truncated, {}
    
    def do_simulation(self, ctrl, n_frames) -> None:
        # if np.array(ctrl).shape != (self.model.nu,):
        #     raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")
        ctrl1 = self._ctbr2srt(ctrl[0:4])
        ctrl2 = self._ctbr2srt(ctrl[4:8])
        ctrl3 = self._ctbr2srt(ctrl[8:12])
        ctrl = np.concatenate([ctrl1, ctrl2, ctrl3])
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self._apply_control(ctrl=ctrl)
        mj.mj_step(self.model, self.data, nstep=n_frames)

    def _apply_control(self, ctrl):
        self.data.actuator("Motor0_1").ctrl[0] = ctrl[0]  # data.ctrl[1] # front
        self.data.actuator("Motor1_1").ctrl[0] = ctrl[1]  # data.ctrl[2] # back
        self.data.actuator("Motor2_1").ctrl[0] = ctrl[2]  # data.ctrl[3] # left
        self.data.actuator("Motor3_1").ctrl[0] = ctrl[3]  # data.ctrl[4] # right

        self.data.actuator("Motor0_2").ctrl[0] = ctrl[4]  # data.ctrl[1] # front
        self.data.actuator("Motor1_2").ctrl[0] = ctrl[5]  # data.ctrl[2] # back
        self.data.actuator("Motor2_2").ctrl[0] = ctrl[6]  # data.ctrl[3] # left
        self.data.actuator("Motor3_2").ctrl[0] = ctrl[7]  # data.ctrl[4] # right

        self.data.actuator("Motor0_3").ctrl[0] = ctrl[8]  # data.ctrl[1] # front
        self.data.actuator("Motor1_3").ctrl[0] = ctrl[9]  # data.ctrl[2] # back
        self.data.actuator("Motor2_3").ctrl[0] = ctrl[10]  # data.ctrl[3] # left
        self.data.actuator("Motor3_3").ctrl[0] = ctrl[11]  # data.ctrl[4] # right

    def _ctbr2srt(self, action):
        zcmd = (self.mQ + self.mP) * self.g * (1 + self.action[0]/2)
        dφd = np.pi/2 * action[1]
        dθd = np.pi/2 * action[2]
        dψd = np.pi/2 * action[3]
        
        ω = self.data.qvel[3:6]
        dω = self.data.qacc[3:6]
        M = self.JQ @ dω + np.cross(ω, self.JQ @ ω)

        self.edφP = dφd - ω[0]
        self.edφI = np.clip(self.edφI + self.edφP * self.sim_dt, -self.clipI, self.clipI)
        self.edφD = (self.edφP - self.edφP_prev) / self.sim_dt
        self.edφP_prev = self.edφP
        φcmd = np.clip(self.kPdφ * self.edφP + self.kIdφ * self.edφI + self.kDdφ * self.edφD, -0.25, 0.25)

        self.edθP = dθd - ω[1]
        self.edθI = np.clip(self.edθI + self.edθP * self.sim_dt, -self.clipI, self.clipI)
        self.edθD = (self.edθP - self.edθP_prev) / self.sim_dt
        self.edθP_prev = self.edθP
        θcmd = np.clip(self.kPdθ * self.edθP + self.kIdθ * self.edθI + self.kDdθ * self.edθD, -0.25, 0.25)

        self.edψP = dψd - ω[2]
        self.edψI = np.clip(self.edψI + self.edψP * self.sim_dt, -self.clipI, self.clipI)
        self.edψD = (self.edψP - self.edψP_prev) / self.sim_dt
        self.edψP_prev = self.edψP
        ψcmd = np.clip(self.kPdψ * self.edψP + self.kIdψ * self.edψI + self.kDdψ * self.edψD, -0.25, 0.25)

        Mcmd = np.array([φcmd, θcmd, ψcmd])
        
        f = self.A @ np.concatenate([[zcmd], Mcmd]).reshape((4,1))
        f = np.clip(f.flatten(), 0, 5)

        return f

    def _update_data(self, reward, step=True):
        if step:
            self.e_buffer_Q1.append(np.copy(self.e_curr_Q1))
            self.e_buffer_Q2.append(np.copy(self.e_curr_Q2))
            self.e_buffer_Q3.append(np.copy(self.e_curr_Q3))
            self.a_buffer_Q1.append(np.copy(self.action[0:4]))
            self.a_buffer_Q2.append(np.copy(self.action[4:8]))
            self.a_buffer_Q3.append(np.copy(self.action[8:12]))

            # Present
            self.time_in_sec = np.round(self.time_in_sec + self.policy_dt, 3)
            self.timestep += 1
            self.total_reward += reward

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def _get_reward(self):
        # total_reward = 0
        # for env in self.random_envs:
        #     reward, _ = env._get_reward()
        #     total_reward += reward
        # return total_reward, {}

        # 1. Get rewards from random envs

        # 2. Return sum and blank info

        return 0, {}
    
    def _terminated(self):
        # terminated = False
        # for env in self.random_envs:
        #     if env._terminated():
        #         self.history_epi[self.traj_type].append(self.timestep)
        #         terminated = True
        # return terminated
        
        # 1. Get termination info from random envs

        # 2. Return &&

        return self.timestep == 2000

    def _truncated(self):
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments')
    
    # Execution parameters
    parser.add_argument('--id', type=str, default='untitled', help='Provide experiment name and ID.')
    parser.add_argument('--visualize', type=bool, default=False, help='Choose visualization option.')
    
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    """ Divide obs into three """
    n_observation = 108
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        obs_Q1 = observations[:,:n_observation]  # .flatten()
        obs_Q2 = observations[:,n_observation:2*n_observation]  # .flatten()
        obs_Q3 = observations[:,2*n_observation:3*n_observation]  # .flatten()
        actions_Q1, states_Q1 = model.predict(
            obs_Q1,  # type: ignore[arg-type]
            state=None,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        actions_Q2, states_Q2 = model.predict(
            obs_Q2,  # type: ignore[arg-type]
            state=None,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        actions_Q3, states_Q3 = model.predict(
            obs_Q3,  # type: ignore[arg-type]
            state=None,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        actions = np.hstack([actions_Q1, actions_Q2, actions_Q3])
        new_observations = np.hstack([states_Q1, states_Q2, states_Q3])
        new_observations, rewards, dones, infos = env.step(actions)  # new_obs = [obs_Q1, obs_Q2, obs_Q3]
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

def main():
    args_dict = parse_arguments()
    print(args_dict)
    
    # Path
    experiment_id = args_dict['id']
    save_path = os.path.join('../train/saved_models/saved_model_'+experiment_id)
    loaded_model = PPO.load(save_path+"/best_model")

    # Environment parameters
    render_mode = 'human' if args_dict['visualize'] else None

    env = QuadrotorMultipleEnv(render_mode=render_mode)
    env = VecMonitor(DummyVecEnv([lambda: env]))
    
    print("Evaluation start!")
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=50, render=render_mode)
    env.close()
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

if __name__ == "__main__":
    main()