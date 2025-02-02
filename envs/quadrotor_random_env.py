"""
    Description:
        Environment for quadrotor payload trajectory tracking
        with random constant force exerted on payload
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
from typing import Dict, Union
from collections import deque
# Gym
from gymnasium.spaces import Box
from gymnasium import utils
# Mujoco
import mujoco as mj
from mujoco_gym.mujoco_env import MujocoEnv
# ETC
from envs.utils.action_filter import ActionFilterButter
from envs.utils.utility_functions import *
import envs.utils.utility_trajectory as ut
from envs.utils.rotation_transformations import *
from envs.utils.geo_tools import hat, vee
from envs.utils.render_util import setup_viewer
import time
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 10.0}
  
class QuadrotorRandomEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps = 10000,  # 2500,
        xml_file: str = "../assets/quadrotor_x_cfg_payload_nominal.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 1.0,
        env_num: int = 0,
        **kwargs
    ):
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)
        self.frame_skip = frame_skip
        
        ##################################################
        #################### DYNAMICS ####################
        ##################################################
        # region
        self.sim_freq: float       = 500.0
        self.policy_freq: float    = 500.0  # 125.0
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
        # endregion
        ##################################################
        #################### BOOLEANS ####################
        ##################################################
        # region
        self.is_future_traj     = False
        self.is_lpf_action      = False  # Low Pass Filter
        self.is_visual          = False
        self.is_launch_control  = False
        self.is_action_bound    = False
        self.is_rs_reward       = False  # Rich-Sutton Reward
        self.is_io_history      = False
        self.is_delayed         = False
        # endregion
        ##################################################
        ################## OBSERVATION ###################
        ##################################################
        # region
        self.env_num            = env_num
        self.n_action           = 4
        self.n_observation      = 108
        self.history_len_short  = 5
        self.history_len_long   = 10
        self.history_len        = self.history_len_short
        self.future_len         = 3
        self.e_buffer           = deque(np.zeros((self.history_len, 6)), maxlen=self.history_len)
        self.a_buffer           = deque(np.zeros((self.history_len_long, 4)), maxlen=self.history_len_long)
        self.a_last             = np.zeros(self.n_action)
        self.num_episode        = 0
        self.history_epi        = {'setpoint': deque([0]*10, maxlen=10),
                                   'curve': deque([0]*10, maxlen=10)}
        self.progress           = {'setpoint': 1e-3,
                                   'curve': 1e-3}
        self.action_space       = self._set_action_space()
        self.observation_space  = self._set_observation_space()
        
        if self.is_delayed:
            self.n_delay        = 3
            self.s_record       = np.zeros((self.max_timesteps, self.s_len))     # For prediction
            self.a_record       = np.zeros((self.max_timesteps, self.n_action))  # For prediction
        else:
            self.n_delay        = 0
        # endregion
        ##################################################
        ##################### BOUNDS #####################
        ##################################################
        # region
        self.pos_err_bound = 0.5
        self.vel_err_bound = 2.0
        # self.pos_err_bound = 10.0
        # self.vel_err_bound = 10.0
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
        self.payload_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'payload')
        # endregion
        ##################################################
        ################# INITIALIZATION #################
        ##################################################
        # region
        self._init_action_filter()
        self._init_env()
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

        self.time = 0

    def _set_action_space(self):
        # CTBR
        low = np.array([-1, -1, -1, -1])
        high = np.array([1, 1, 1, 1])
        if self.is_action_bound: self.action_space = Box(low=0.2*np.ones(4), high=0.8*np.ones(4))
        else: self.action_space = Box(low=low, high=high)
        return self.action_space

    def _set_observation_space(self):
        if self.is_io_history: obs_shape = 40
        else: obs_shape = self.n_observation
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        return observation_space

    def _init_env(self):
        print("Environment {} created".format(self.env_num))
        print("Sample action: {}".format(self.action_space.sample()))
        print("Action Space: {}".format(np.array(self.action_space)))
        print("Observation Space: {}".format(np.array(self.observation_space)))
        print("Timestep: {}".format(self.policy_dt))
        print("-"*100)

    def _init_action_filter(self):
        self.action_filter = ActionFilterButter(
            lowcut        = [0.0],
            highcut       = [0.4 * self.policy_freq],
            sampling_rate = self.policy_freq,
            order         = 2,
            num_joints    = self.n_action,
        )

    def _init_history_ff(self):
        [self.e_buffer.append(np.copy(self.e_curr)) for _ in range(self.history_len)]
        [self.a_buffer.append(np.copy(self.action)) for _ in range(self.history_len)]

    def reset(self, seed=None, randomize=None):
        # super().reset(seed=self.env_num)
        self.action = np.zeros(self.n_action)
        self.e_curr = np.zeros(6)
        self.a_last = np.zeros(self.n_action)
        self.q_last = np.array([0,0,-1])

        self.action_filter.reset()
        self.action_filter.init_history(np.zeros(self.n_action))
        self._init_history_ff()

        self._reset_env()
        self._reset_model()
        self._reset_error()
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
        """ Compute progress """
        self.progress['setpoint'] = (np.mean(self.history_epi['setpoint']) / self.max_timesteps)
        self.progress['curve'] = (np.mean(self.history_epi['curve']) / self.max_timesteps)
        
        """ TEST """
        self.progress['setpoint'] = 1.0
        self.progress['curve'] = 0.5

        """ Choose trajectory type """
        if self.progress['setpoint'] < 0.5:
            self.stage = 1
            self.traj_type = np.random.choice(['setpoint', 'curve'], p=[0.9, 0.1])
        else:
            self.stage = 2
            self.traj_type = np.random.choice(['setpoint', 'curve'], p=[0.1, 0.9])

        """ Generate trajectory """
        self._generate_trajectory()

        """ Randomize Inertia """
        self._randomize_inertia()

        """ Initial Position """
        rot = euler2rot(np.array([0, 0, np.random.uniform(0, 2*np.pi)]))
        self.xPr = rot @ np.array([0.5, -0.289, -0.866])
        self.xN1r = rot @ np.array([1, 0, 0])
        self.xN2r = rot @ np.array([0.5, -0.866, 0])

        xQ0 = self.xQd[0]
        xP0 = xQ0 + self.xPr
        self.xN10 = xQ0 + self.xN1r
        self.xN20 = xQ0 + self.xN2r

        vQ0 = self.vQd[0]
        vP0 = self.vQd[0]
        vN10 = self.vQd[0]
        vN20 = self.vQd[0]
        
        qpos = np.concatenate([xQ0, self.init_qpos[3:7], self.init_qpos[7:11], xP0, self.init_qpos[14:18], self.init_qpos[18:22], self.xN10, self.init_qpos[25:33], self.xN20, self.init_qpos[36:44]])
        qvel = np.concatenate([vQ0, self.init_qvel[3:6], self.init_qvel[6:9], vP0, self.init_qvel[12:15], self.init_qvel[15:18], vN10, self.init_qvel[21:27], vN20, self.init_qvel[30:36]])
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _reset_renderer(self):
        self.render()
        setup_viewer(self.mujoco_renderer.viewer)
        del self.mujoco_renderer.viewer._markers[:]

    def _reset_error(self):
        self.edφI = 0
        self.edθI = 0
        self.edψI = 0

        self.edφP_prev = 0
        self.edθP_prev = 0
        self.edψP_prev = 0

    def _generate_trajectory(self):
        """ Set trajectory parameters """
        if self.traj_type == 'setpoint':
            self.traj = ut.CrazyTrajectory(tf=self.max_timesteps*self.policy_dt, ax=0, ay=0, az=0, f1=0, f2=0, f3=0)
            self.difficulty = self.stage * self.progress["setpoint"]
        if self.traj_type == 'curve':
            self.traj = ut.CrazyTrajectory(tf=self.max_timesteps*self.policy_dt,
                                           ax=np.random.choice([-1,1])*3*self.progress["curve"],
                                           ay=np.random.choice([-1,1])*3*self.progress["curve"],
                                           az=np.random.choice([-1,1])*3*self.progress["curve"],
                                           f1=np.random.choice([-1,1])*0.4*self.progress["curve"],
                                           f2=np.random.choice([-1,1])*0.4*self.progress["curve"],
                                           f3=np.random.choice([-1,1])*0.4*self.progress["curve"],
                                          )
            self.difficulty = self.stage * self.progress["curve"]

        # self.traj.plot()
        # self.traj.plot3d_payload()

        """ Compute trajectory """
        self.xQd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        self.vQd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        for i in range(self.n_delay):
            self.xQd[i], self.vQd[i], _ = self.traj.get(0)
        for i in range(self.n_delay, self.max_timesteps + self.history_len + self.n_delay):
            self.xQd[i], self.vQd[i], _ = self.traj.get((i - self.n_delay)*self.policy_dt)
        self.goal_pos = self.xQd[-1]

    def _randomize_inertia(self):
        self.inertia_randomness = 0.1
        self.body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "quadrotor")
        self.model.body_inertia[self.body_id] = np.array([0.49, 0.53, 0.98]) * 1e-2 * np.clip(np.random.normal(1, 0.1, 3), -0.9, 1.1)
        self.JQ = np.diag(self.model.body_inertia[self.body_id])

    def _get_obs(self):
        # Present
        self.obs_curr = self._get_obs_curr()

        # Past
        e_buffer = np.array(self.e_buffer).flatten()  # 30
        a_buffer = np.array(self.a_buffer)[-self.history_len:,:].flatten()  # 20
        io_history = np.concatenate([e_buffer, a_buffer])  # 50

        # Future
        xQ_ff = self.xQ - self.xQd[self.timestep + 1 : self.timestep + self.future_len + 1]
        vQ_ff = self.vQ - self.vQd[self.timestep + 1 : self.timestep + self.future_len + 1]
        ff = np.concatenate([(xQ_ff @ self.RQ).flatten(),
                             (vQ_ff @ self.RQ).flatten()])  # 18

        obs_full = np.concatenate([self.obs_curr, io_history, ff])  # 40+50+18 = 108

        return obs_full

    def _get_obs_curr(self):
        self.xQ = self.data.qpos[0:3]
        self.xP = self.data.qpos[11:14]
        self.xN1 = self.data.qpos[22:25]
        self.xN2 = self.data.qpos[33:36]
        self.vQ = self.data.qvel[0:3]
        self.vP = self.data.qvel[9:12]
        self.vN1 = self.data.qvel[18:21]
        self.vN2 = self.data.qvel[27:30]
        self.RQ = quat2rot(self.data.qpos[3:7])  # Quadrotor
        self.ωQ = self.data.qvel[3:6]  # Quadrotor
        
        self.exQ = self.xQ - self.xQd[self.timestep]
        self.evQ = self.vQ - self.vQd[self.timestep]

        self.e_curr = np.concatenate([self.RQ.T @ self.exQ, self.RQ.T @ self.evQ])
        obs_curr = np.concatenate([self.e_curr,
                                   self.RQ.T @ (self.xQ - self.xP), self.RQ.T @ (self.vQ - self.vP), 
                                   self.RQ.T @ (self.xQ - self.xN1), self.RQ.T @ (self.vQ - self.vN1),
                                   self.RQ.T @ (self.xQ - self.xN2), self.RQ.T @ (self.vQ - self.vN2),
                                   self.action, self.RQ.flatten(), self.ωQ])  # 40
        
        return obs_curr

    def step(self, action, restore=False):
        if self.traj_type == 'curve':
            self.data.qpos[0:3] = self.xQd[self.timestep]
            self.data.qpos[22:25] = self.xQd[self.timestep] + self.xN1r 
            self.data.qpos[33:36] = self.xQd[self.timestep] + self.xN2r 
            self.data.qvel[0:3] = self.vQd[self.timestep]
            self.data.qvel[18:21] = self.vQd[self.timestep]
            self.data.qvel[27:30] = self.vQd[self.timestep]
        else:
            self.data.qpos[22:25] = self.xN10
            self.data.qpos[33:36] = self.xN20
            self.data.qvel[18:21] = np.zeros(3)
            self.data.qvel[27:30] = np.zeros(3)
        self.action = action
        for _ in range(self.num_sims_per_env_step):
            self.do_simulation(self.action, self.frame_skip)  # a_{t}
        if self.render_mode == 'human':
            self.render()
        obs_full = self._get_obs()
        reward, reward_dict = self._get_reward()
        self.info["reward_dict"] = reward_dict
        terminated = self._terminated()
        truncated = self._truncated()
        self._update_data(reward=reward, step=True)

        return obs_full, reward, terminated, truncated, {}
    
    def do_simulation(self, ctrl, n_frames) -> None:
        # if np.array(ctrl).shape != (self.model.nu,):
        #     raise ValueError(f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}")
        ctrl = self._ctbr2srt(ctrl)
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self._apply_control(ctrl=ctrl)
        mj.mj_step(self.model, self.data, nstep=n_frames)

    def _apply_control(self, ctrl):
        self.data.actuator("Motor0").ctrl[0] = ctrl[0]  # data.ctrl[1] # front
        self.data.actuator("Motor1").ctrl[0] = ctrl[1]  # data.ctrl[2] # back
        self.data.actuator("Motor2").ctrl[0] = ctrl[2]  # data.ctrl[3] # left
        self.data.actuator("Motor3").ctrl[0] = ctrl[3]  # data.ctrl[4] # right

    def _ctbr2srt(self, action):
        zcmd = (self.mQ + self.mP) * self.g * (1 + self.action[0]/2)
        dφd = np.pi/2 * action[1]
        dθd = np.pi/2 * action[2]
        dψd = np.pi/2 * action[3]

        # zcmd = self.mQ * self.g * (1 + 0.5 * self.action[0])
        # dφd = 0.5 * np.pi * action[1]
        # dθd = 0.5 * np.pi * action[2]
        # dψd = 0.5 * np.pi * action[3]
        # zcmd = 8
        # dφd = 0
        # dθd = 0
        # dψd = 0
        
        ω = self.data.qvel[3:6]
        dω = self.data.qacc[3:6]
        M = self.JQ @ dω + np.cross(ω, self.JQ @ ω)
        # print("ω: ", np.round(ω,2))
        # print("dω: ", np.round(dω,2))
        # print("M: ", np.round(M,2))
        # print("Quad qvel:", np.round(self.data.qvel[0:6],2))

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

        # print("M: ", M)
        # print("Md: ", np.round(Md, 2))
        # print("f: ", np.round(f, 2))
        # print("zcmd: ", np.round(zcmd, 2))
        # print("φcmd: ", np.round(φcmd, 2))
        # print("θcmd: ", np.round(θcmd, 2))
        # print("ψcmd: ", np.round(ψcmd, 2))
        # print("P: ", np.round(self.kPdφ * self.edφP, 2))
        # print("I: ", np.round(self.kIdφ * self.edφI, 2))
        # print("D: ", np.round(self.kDdφ * self.edφD, 2))

        return f

    def _update_data(self, reward, step=True):
        if step:
            self.e_buffer.append(np.copy(self.e_curr))
            self.a_buffer.append(np.copy(self.action))
            self.a_last = np.copy(self.action)

            # Present
            self.time_in_sec = np.round(self.time_in_sec + self.policy_dt, 3)
            self.timestep += 1
            self.total_reward += reward
    
    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def _render_marker(self):
        marker_pos = self.xP + self.direction
        self.mujoco_renderer.viewer.add_marker(pos=marker_pos, size=np.array([0.05, 0.05, 0.05]), rgba=np.array([1, 0, 0, 1]))

    def _get_reward(self):
        names = ['xQ_rew', 'vQ_rew', 'ψQ_rew', 'ωQ_rew']
        
        w_xQ = 1.0
        w_vQ = 0.25
        w_ψQ = 0.5
        w_ωQ = 0.25

        reward_weights = np.array([w_xQ, w_vQ, w_ψQ, w_ωQ])
        weights = reward_weights / np.sum(reward_weights)

        scale_xQ = 1.0/0.5
        scale_vQ = 1.0/2.0
        scale_ψQ = 1.0/(np.pi/2)
        scale_ωQ = 1.0/(np.pi/2)

        ψQ  = quat2euler_raw(self.data.qpos[3:7])[2]
        
        exQ = np.linalg.norm(self.exQ, ord=2)
        evQ = np.linalg.norm(self.evQ, ord=2)
        eψQ = np.abs(ψQ)
        eωQ = np.linalg.norm(self.ωQ, ord=2)

        rewards = np.exp(-np.array([scale_xQ, scale_vQ, scale_ψQ, scale_ωQ])
                         *np.array([exQ, evQ, eψQ, eωQ]))
        reward_dict = dict(zip(names, weights * rewards))
        
        if self.traj_type == "setpoint":
            total_reward = np.sum(weights * rewards)
        else:
            total_reward = np.sum(weights * rewards) * (1 + 0.5 * self.difficulty)

        return total_reward, reward_dict

    def _terminated(self):
        attQ = quat2euler_raw(self.data.qpos[3:7])
        xQd = self.xQd[self.timestep] if self.timestep < self.max_timesteps else self.goal_pos
        vQd = self.vQd[self.timestep] if self.timestep < self.max_timesteps else np.zeros(3)
        exQ = np.linalg.norm(self.xQ - xQd)
        evQ = np.linalg.norm(self.vQ - vQd)

        if exQ > self.pos_err_bound:
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Pos error: {pos_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(np.round(self.difficulty, 2)),
                  pos_err=np.round(exQ,1),
                  time=np.round(self.timestep*self.policy_dt,2),
                  rew=np.round(self.total_reward,1)))
            return True
        elif evQ > self.vel_err_bound:
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Vel error: {vel_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(np.round(self.difficulty, 2)),
                  vel_err=np.round(evQ,1),
                  time=np.round(self.timestep*self.policy_dt,2),
                  rew=np.round(self.total_reward,1)))
            return True
        elif not(np.abs(attQ) < np.pi/2).all():
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Att error: {att} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(np.round(self.difficulty, 2)),
                  att=np.round(attQ,1),
                  time=np.round(self.timestep*self.policy_dt,2),
                  rew=np.round(self.total_reward,1)))
            return True
        elif self.timestep >= self.max_timesteps:
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Max time: {time} | Final pos error: {pos_err} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(np.round(self.difficulty, 2)),
                  time=np.round(self.timestep*self.policy_dt,2),
                  pos_err=np.round(exQ,2),
                  rew=np.round(self.total_reward,1)))
            return True
        else:
            return False

    def _truncated(self):
        return False

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()


if __name__ == "__main__":
    env = QuadrotorRandomEnv()