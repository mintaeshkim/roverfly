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
import time
import matplotlib.pyplot as plt
import pandas as pd



DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 10.0}
  
class QuadrotorPayloadEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps = 2000,
        xml_file: str = "../assets/quadrotor_x_cfg_payload.xml",
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
        self.policy_freq: float    = 100.0
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
        self.n_state            = 18  # xQ, ΘQ, xP, vQ, ωQ, vP
        self.n_action           = 4
        self.n_observation      = 186
        self.history_len_short  = 5
        self.history_len_long   = 10
        self.history_len        = self.history_len_short
        self.future_len         = 3
        self.s_buffer           = deque(np.zeros((self.history_len_long, self.n_state)), maxlen=self.history_len_long)
        self.e_buffer           = deque(np.zeros((self.history_len, 12)), maxlen=self.history_len)
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
            self.s_record       = np.zeros((self.max_timesteps, self.n_state))
            self.a_record       = np.zeros((self.max_timesteps, self.n_action))
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
        self.e_curr = np.zeros(12)
        self.a_last = np.zeros(self.n_action)
        self.q_last = np.array([0,0,-1])

        self.action_filter.reset()
        self.action_filter.init_history(np.zeros(self.n_action))
        self._init_history_ff()

        self._reset_env()
        self._reset_model()
        self._reset_error()

        obs = self._get_obs()
        self.info = self._get_reset_info()

        return obs, self.info
    
    def _reset_env(self):
        self.timestep     = 0  # discrete timestep, k
        self.time_in_sec  = 0.0  # time
        self.total_reward = 0
        self.terminated   = None
        self.info         = {}

        """ TEST """
        # self.test_record_P = TestRecord(max_timesteps=self.max_timesteps,
        #                                 record_object='P',
        #                                 num_sims_per_env_step=self.num_sims_per_env_step)
        # self.test_record_Q = TestRecord(max_timesteps=self.max_timesteps,
        #                                 record_object='Q',
        #                                 num_sims_per_env_step=self.num_sims_per_env_step)

    def _reset_model(self):
        """ Compute progress """
        self.progress['setpoint'] = (np.mean(self.history_epi['setpoint']) / self.max_timesteps)
        self.progress['curve'] = (np.mean(self.history_epi['curve']) / self.max_timesteps)
        
        """ TEST """
        # self.progress['setpoint'] = 1.0
        # self.progress['curve'] = 0.6

        """ Choose trajectory type """
        if self.progress['setpoint'] < 0.5:
            self.stage = 1
            self.traj_type = np.random.choice(['setpoint', 'curve'], p=[0.9, 0.1])
        else:
            self.stage = 2
            self.traj_type = np.random.choice(['setpoint', 'curve'], p=[0.1, 0.9])

        # self.traj_type = 'setpoint'
        # self.traj_type = 'curve'

        """ Set trajectory parameters """
        if self.traj_type == 'setpoint':
            self.traj = ut.CrazyTrajectoryPayload(tf=self.max_timesteps*self.policy_dt, ax=0, ay=0, az=0, f1=0, f2=0, f3=0)
            self.difficulty = self.stage * self.progress["setpoint"]
        if self.traj_type == 'curve':
            self.traj = ut.CrazyTrajectoryPayload(tf=self.max_timesteps*self.policy_dt,
                                                  ax=np.random.choice([-1,1])*3*self.progress["curve"],
                                                  ay=np.random.choice([-1,1])*3*self.progress["curve"],
                                                  az=np.random.choice([-1,1])*3*self.progress["curve"],
                                                  f1=np.random.choice([-1,1])*0.4*self.progress["curve"],
                                                  f2=np.random.choice([-1,1])*0.4*self.progress["curve"],
                                                  f3=np.random.choice([-1,1])*0.4*self.progress["curve"],
                                                #   ax=2, ay=2, az=0, f1=0.4, f2=0.4, f3=0
                                                  )
            # self.traj = ut.SmoothTraj5Payload(x0=np.array([0, 0, 0.5]), xf=np.array([0, 0, 1]), tf=5)

            self.difficulty = self.stage * self.progress["curve"]

        # self.traj.plot()
        # self.traj.plot3d_payload()

        """ Compute trajectory """
        self.xPd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        self.vPd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        self.aPd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        self.daPd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        self.qd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        self.dqd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        self.d2qd = np.zeros((self.max_timesteps + self.history_len + self.n_delay, 3))
        for i in range(self.n_delay):
            self.xPd[i], self.vPd[i], self.aPd[i], self.daPd[i], self.qd[i], self.dqd[i], self.d2qd[i] = self.traj.get(0)
        for i in range(self.n_delay, self.max_timesteps + self.history_len + self.n_delay):
            self.xPd[i], self.vPd[i], self.aPd[i], self.daPd[i], self.qd[i], self.dqd[i], self.d2qd[i] = self.traj.get((i - self.n_delay)*self.policy_dt)
        self.goal_pos = self.xPd[-1]
        self.dqd[0], self.d2qd[0] = np.zeros(3), np.zeros(3)

        """ Randomize Inertia """
        self.inertia_randomness = 0.1
        self.body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "quadrotor")
        self.model.body_inertia[self.body_id] = np.array([0.49, 0.53, 0.98]) * 1e-2 * np.clip(np.random.normal(1, 0.1, 3), -0.9, 1.1)
        self.JQ = np.diag(self.model.body_inertia[self.body_id])

        """ Initial Perturbation """
        # region
        # No perturbation in attitude, velocity, and angular velocity
        if self.traj_type == 'setpoint':
            self.perturbation = self.progress["setpoint"]
        else:
            self.perturbation = 1e-5
        # self.perturbation = self.progress["curve"]

        watt = self._reset_noise_scale * (np.pi/18) * np.random.uniform(0, self.perturbation)
        wv = self._reset_noise_scale * 0.1 * np.random.uniform(0, self.perturbation)
        wω = self._reset_noise_scale * (np.pi/36) * np.random.uniform(0, self.perturbation)

        dPQ = 1.0 - 0.01 * np.random.uniform(0, self.perturbation)
        psi = np.random.uniform(0, 2*np.pi)
        phi = (np.pi/4) * np.clip(np.random.normal(0, self.perturbation), -0.5, 0.5)
        # phi = np.pi/8
        
        xP = self.xPd[0]
        xQ = xP + dPQ * np.array([np.cos(psi)*np.sin(phi), np.sin(psi)*np.sin(phi), np.cos(phi)])
        attP = euler2quat_raw(quat2euler_raw(self.init_qpos[14:18]) + self.np_random.uniform(size=3, low=-watt, high=watt))
        attQ = euler2quat_raw(quat2euler_raw(self.init_qpos[3:7]) + self.np_random.uniform(size=3, low=-watt, high=watt))
        vP = self.vPd[0] + self.np_random.uniform(size=3, low=-wv, high=wv)
        vQ = self.vPd[0] + self.np_random.uniform(size=3, low=-wv, high=wv)
        ωP = self.init_qvel[12:15] + self.np_random.uniform(size=3, low=-wω, high=wω)
        ωQ = self.init_qvel[3:6] + self.np_random.uniform(size=3, low=-wω, high=wω)
        # endregion
        
        qpos = np.concatenate([xQ, attQ, self.init_qpos[7:11], xP, attP, self.init_qpos[18:22]])
        qvel = np.concatenate([vQ, ωQ, self.init_qvel[6:9], vP, ωP, self.init_qvel[15:18]])
        self.set_state(qpos, qvel)

        if self.is_delayed:
            data_curr = np.concatenate([np.copy(self.data.qpos[0:3]), quat2euler_raw(np.copy(self.data.qpos[3:7])), np.copy(self.data.qpos[11:14]), np.copy(self.data.qvel[0:6]), np.copy(self.data.qvel[9:12])])
            [self.s_buffer.append(data_curr) for _ in range(self.history_len_long)]

        return self._get_obs()

    def _reset_error(self):
        self.edφI = 0
        self.edθI = 0
        self.edψI = 0

        self.edφP_prev = 0
        self.edθP_prev = 0
        self.edψP_prev = 0

    def _get_obs(self):
        # Present
        self.obs_curr = self._get_obs_curr()

        # Past
        e_buffer = np.array(self.e_buffer, dtype=object).flatten()  # 60
        a_buffer = np.array(self.a_buffer, dtype=object)[-self.history_len:,:].flatten()  # 20
        s_buffer = np.array(self.s_buffer, dtype=object)[-self.history_len:,:]
        Θ_buffer = np.array([s_buffer[i,3:6] for i in range(self.history_len)]).flatten()  # 15
        ω_buffer = s_buffer[:,12:15].flatten()  # 15
        io_history = np.concatenate([e_buffer, a_buffer, Θ_buffer, ω_buffer])  # 110

        # Future
        xP_ff = self.xP - self.xPd[self.timestep : self.timestep + self.future_len]
        vP_ff = self.vP - self.vPd[self.timestep : self.timestep + self.future_len]
        xQd = self.xPd[self.timestep : self.timestep + self.future_len] - self.qd[self.timestep : self.timestep + self.future_len]
        vQd = self.vPd[self.timestep : self.timestep + self.future_len] - self.dqd[self.timestep : self.timestep + self.future_len]
        xQ_ff = self.xQ - xQd
        vQ_ff = self.vQ - vQd
        ff = np.concatenate([(xP_ff @ self.R).flatten(),
                             (vP_ff @ self.R).flatten(),
                             (xQ_ff @ self.R).flatten(),
                             (vQ_ff @ self.R).flatten()])  # 36

        obs_full = np.concatenate([self.obs_curr, io_history, ff])  # 40+110+36 = 186

        return obs_full

    def _get_obs_curr(self):
        if self.is_delayed:
            # NOTE: Use delayed data
            qpos = self.s_buffer[-self.n_delay][:9]  # xQ, ΘQ, xP
            qvel = self.s_buffer[-self.n_delay][9:]  # vQ, ωQ, vP

            # NOTE: Use fitted data
            # region
            # self.s_predicted = self.predict()
            # qpos = self.s_predicted[:9]
            # qvel = self.s_predicted[9:]

            # print("Pos")
            # print("xQ Error: ", np.round(qpos[:3] - self.data.qpos[0:3], 3))
            # print("xP Error: ", np.round(qpos[6:] - self.data.qpos[11:14], 3))
            # print("Vel")
            # print("vQ Error: ", np.round(qvel[:3] - self.data.qvel[0:3], 3))
            # print("vP Error: ", np.round(qvel[6:] - self.data.qvel[9:12], 3))
            # endregion
        else:
            qpos = np.concatenate([self.data.qpos[0:3], quat2euler_raw(self.data.qpos[3:7]), self.data.qpos[11:14]])
            qvel = np.concatenate([self.data.qvel[0:6], self.data.qvel[9:12]])

        # NOTE: Record
        # qpos = np.concatenate([self.data.qpos[0:3], quat2euler_raw(self.data.qpos[3:7]), self.data.qpos[11:14]])
        # qvel = np.concatenate([self.data.qvel[0:6], self.data.qvel[9:12]])
        # self.s_record[self.timestep-1] = np.concatenate([qpos, qvel])
        # self.a_record[self.timestep-1] = self.action

        self.xQ = qpos[:3]
        self.xP = qpos[6:]
        self.vQ = qvel[:3]
        self.vP = qvel[6:]
        self.R = euler2rot(qpos[3:6])  # Quadrotor
        self.ω = qvel[3:6]  # Quadrotor
        
        self.exP = self.xP - self.xPd[self.timestep]
        self.evP = self.vP - self.vPd[self.timestep]
        
        self.q = (self.xP - self.xQ) / np.linalg.norm(self.xP - self.xQ)
        self.dq = (self.q - self.q_last) / self.policy_dt
        self.q_last = self.q
        qd, dqd = self._get_qd()

        self.xQd = self.xPd[self.timestep] - qd
        self.vQd = self.vPd[self.timestep] - dqd

        self.exQ = self.xQ - self.xQd
        self.evQ = self.vQ - self.vQd

        # self.eq = hat(self.q)**2 @ self.qd[self.timestep]
        # self.edq = self.dq - np.cross(np.cross(self.qd[self.timestep], self.dqd[self.timestep]), self.q)

        self.e_curr = np.concatenate([self.R.T @ self.exP, self.R.T @ self.evP, self.R.T @ self.exQ, self.R.T @ self.evQ])

        obs_curr = np.concatenate([self.e_curr,                                                           # 12
                                   self.R.T @ self.q, self.R.T @ self.dq, self.R.T @ qd, self.R.T @ dqd,  # 12
                                   self.action, self.R.flatten(), self.ω])                                # 16
        return obs_curr

    # NOTE: System ID
    def predict(self):
        s_buffer = np.array(self.s_buffer).T  # 18:H
        a_buffer = np.array(self.a_buffer).T  # 4:H
        
        s_next = s_buffer[:, 1:-self.n_delay+1]  # 18x21
        s_curr = s_buffer[:, :-self.n_delay]  # 18x21
        a_curr = a_buffer[:, :-self.n_delay]  # 4x21
        s_a_curr = np.concatenate([s_curr, a_curr], axis=0)  # 22x21

        beta = s_next @ np.linalg.pinv(s_a_curr)  # s_{k+1} = A s_{k} + B a_{k}
        A = beta[:, :self.n_state]  # A: 18x18
        B = beta[:, self.n_state:]  # B: 18x4

        s_predicted = s_next  # s_{t-3}
        for i in range(self.n_delay):
            a_curr = a_buffer[:, i+1:i+(self.history_len_long-self.n_delay)+1]  # a_{t-3}
            s_predicted = A @ s_predicted + B @ a_curr  # s_{t-2} -> s_{t-1} -> s_{t}

        return s_predicted[:, -1].flatten()  # s_{t}

    def _get_qd(self):
        l = 1.0
        kx = 2 * np.diag([0.5, 0.5, 0.5])
        kv = 2 * np.diag([0.75, 0.75, 0.75])
        aPd = self.aPd[self.timestep] if self.timestep < self.max_timesteps else np.zeros(3)
        dqd = self.dqd[self.timestep] if self.timestep < self.max_timesteps else self.dqd[-1]

        # Compute qd that changes through time
        Fff = (self.mQ + self.mP) * (aPd + np.array([0,0,self.g])) + self.mQ * l * np.dot(self.dq, self.dq) * self.q
        Fpd = - kx @ self.exP - kv @ self.evP
        A = Fff + Fpd
        qd = - A / np.linalg.norm(A)

        return qd, dqd

    def step(self, action, restore=False):
        # 1. Simulate for Single Time Step
        self.action = action
        for _ in range(self.num_sims_per_env_step):
            self.do_simulation(self.action, self.frame_skip)  # a_{t}
        if self.render_mode == "human": self.render()
        # 2. Get Observation
        obs_full = self._get_obs()
        # 3. Get Reward
        reward, reward_dict = self._get_reward()
        self.info["reward_dict"] = reward_dict
        # 4. Termination / Truncation
        terminated = self._terminated()
        truncated = self._truncated()
        # 5. Update Data
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
        zcmd = (self.mQ + self.mP) * self.g * (action[0] + 1) / 2
        dφd = action[1]
        dθd = action[2]
        dψd = action[3]

        self.edφP = dφd - self.ω[0]
        self.edφI = np.clip(self.edφI + self.edφP * self.sim_dt, -self.clipI, self.clipI)
        self.edφD = (self.edφP - self.edφP_prev) / self.sim_dt
        self.edφP_prev = self.edφP
        φcmd = np.clip(self.kPdφ * self.edφP + self.kIdφ * self.edφI + self.kDdφ * self.edφD, -1, 1)

        self.edθP = dθd - self.ω[1]
        self.edθI = np.clip(self.edθI + self.edθP * self.sim_dt, -self.clipI, self.clipI)
        self.edθD = (self.edθP - self.edθP_prev) / self.sim_dt
        self.edθP_prev = self.edθP
        θcmd = np.clip(self.kPdθ * self.edθP + self.kIdθ * self.edθI + self.kDdθ * self.edθD, -1, 1)

        self.edψP = dψd - self.ω[2]
        self.edψI = np.clip(self.edψI + self.edψP * self.sim_dt, -self.clipI, self.clipI)
        self.edψD = (self.edψP - self.edψP_prev) / self.sim_dt
        self.edψP_prev = self.edψP
        ψcmd = np.clip(self.kPdψ * self.edψP + self.kIdψ * self.edψI + self.kDdψ * self.edψD, -1, 1)

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
            """ TEST """
            # self._record()
            # Past
            if self.is_delayed:
                xQ = self.data.qpos[0:3] + np.clip(np.random.normal(0, 0.01, 3), -0.01, 0.01)
                ΘQ = quat2euler_raw(self.data.qpos[3:7]) + np.clip(np.random.normal(0, 0.01, 3), -0.01, 0.01)
                xP = self.data.qpos[11:14] + np.clip(np.random.normal(0, 0.01, 3), -0.01, 0.01)
                vQ = self.data.qvel[0:3] + np.clip(np.random.normal(0, 0.05, 3), -0.05, 0.05)
                ωQ = self.data.qvel[3:6] + np.clip(np.random.normal(0, 0.05, 3), -0.05, 0.05)
                vP = self.data.qvel[9:12] + np.clip(np.random.normal(0, 0.05, 3), -0.05, 0.05)
                s_curr = np.concatenate([xQ, ΘQ, xP, vQ, ωQ, vP])
                self.s_buffer.append(s_curr)
            
            self.e_buffer.append(np.copy(self.e_curr))
            self.a_buffer.append(np.copy(self.action))
            self.a_last = np.copy(self.action)

            # Present
            self.time_in_sec = np.round(self.time_in_sec + self.policy_dt, 3)
            self.timestep += 1
            self.total_reward += reward
    
    def _record(self):
        self.test_record_P.record(pos_curr=self.xP, vel_curr=self.vP, pos_d=self.xPd[self.timestep], vel_d=self.vPd[self.timestep])
        self.test_record_Q.record(pos_curr=self.xQ, vel_curr=self.vQ, pos_d=self.xQd, vel_d=self.vQd)

    def _get_reward(self):
        names = ['xP_rew', 'vP_rew', 'xQ_rew', 'vQ_rew', 'ψQ_rew', 'a_rew']
        
        w_xP = 1.0
        w_vP = 0.5
        w_xQ = 0.8
        w_vQ = 0.4
        w_ψQ = 0.8
        w_ωQ = 0.5

        reward_weights = np.array([w_xP, w_vP, w_xQ, w_vQ, w_ψQ, w_ωQ])
        weights = reward_weights / np.sum(reward_weights)

        scale_xP = 1.0/0.5
        scale_vP = 1.0/2.0
        scale_xQ = 1.0/0.5
        scale_vQ = 1.0/2.0
        scale_ψQ = 1.0/(np.pi/2)
        scale_ωQ = 1.0/1.0

        ψQd = 0
        ψQ  = quat2euler_raw(self.data.qpos[3:7])[2]
        
        exP = np.linalg.norm(self.exP, ord=2)
        evP = np.linalg.norm(self.evP, ord=2)
        exQ = np.linalg.norm(self.exQ, ord=2)
        evQ = np.linalg.norm(self.evQ, ord=2)
        eψQ = np.abs(ψQ - ψQd)
        eωQ = np.linalg.norm(self.ω, ord=2)

        rewards = np.exp(-np.array([scale_xP, scale_vP, scale_xQ, scale_vQ, scale_ψQ, scale_ωQ])
                         *np.array([exP, evP, exQ, evQ, eψQ, eωQ]))
        reward_dict = dict(zip(names, weights * rewards))
        
        if self.traj_type == "setpoint":
            total_reward = np.sum(weights * rewards)
        else:
            total_reward = np.sum(weights * rewards) * (1 + 0.5 * self.difficulty)

        return total_reward, reward_dict

    def _terminated(self):
        xP = self.data.qpos[11:14]
        vP = self.data.qvel[9:12]
        attQ = quat2euler_raw(self.data.qpos[3:7])

        xPd = self.xPd[self.timestep] if self.timestep < self.max_timesteps else self.goal_pos
        vPd = self.vPd[self.timestep] if self.timestep < self.max_timesteps else np.zeros(3)
        exP = np.linalg.norm(xP - xPd)
        evP = np.linalg.norm(vP - vPd)

        if exP > self.pos_err_bound:
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Pos error: {pos_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(np.round(self.difficulty, 2)),
                  pos_err=np.round(exP,1),
                  time=np.round(self.timestep*self.policy_dt,2),
                  rew=np.round(self.total_reward,1)))
            return True
        elif evP > self.vel_err_bound:
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Vel error: {vel_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(np.round(self.difficulty, 2)),
                  vel_err=np.round(evP,1),
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
            """ TEST """
            # self.test_record_P.plot_error()
            # self.test_record_Q.plot_error()
            # self.test_record_P.save_data()
            # self.test_record_Q.save_data()
            # self.test_record_P.reset()
            # self.test_record_Q.reset()
            
            # NOTE: Create a pandas DataFrame from the dictionary
            # s_df = pd.DataFrame(self.s_record)
            # s_df.to_csv('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_{type}.csv'.format(type=self.traj_type + "_" + str(np.round(self.difficulty, 2))), index=False)
            # print("Data saved to s_record_{type}.csv".format(type=self.traj_type + "_" + str(np.round(self.difficulty, 2))))
            # a_df = pd.DataFrame(self.a_record)
            # a_df.to_csv('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_{type}.csv'.format(type=self.traj_type + "_" + str(np.round(self.difficulty, 2))), index=False)
            # print("Data saved to a_record_{type}.csv".format(type=self.traj_type + "_" + str(np.round(self.difficulty, 2))))
            
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Max time: {time} | Final pos error: {pos_err} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(np.round(self.difficulty, 2)),
                  time=np.round(self.timestep*self.policy_dt,2),
                  pos_err=np.round(exP,2),
                  rew=np.round(self.total_reward,1)))
            return True
        else:
            return False

    def _truncated(self):
        return False

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()



class TestRecord:
    def __init__(self, max_timesteps, record_object, num_sims_per_env_step):
        self.max_timesteps = max_timesteps
        self.num_sims_per_env_step = num_sims_per_env_step
        
        self.max_steps = self.max_timesteps // self.num_sims_per_env_step
        self.step = 0
        self.rec_obj = record_object

        self.pos_x = np.zeros(self.max_steps)
        self.pos_y = np.zeros(self.max_steps)
        self.pos_z = np.zeros(self.max_steps)
        self.pos_x_d = np.zeros(self.max_steps)
        self.pos_y_d = np.zeros(self.max_steps)
        self.pos_z_d = np.zeros(self.max_steps)

        self.vel_x = np.zeros(self.max_steps)
        self.vel_y = np.zeros(self.max_steps)
        self.vel_z = np.zeros(self.max_steps)
        self.vel_x_d = np.zeros(self.max_steps)
        self.vel_y_d = np.zeros(self.max_steps)
        self.vel_z_d = np.zeros(self.max_steps)
    
    def record(self, pos_curr, vel_curr, pos_d, vel_d):
        self.pos_x[self.step] = pos_curr[0]
        self.pos_y[self.step] = pos_curr[1]
        self.pos_z[self.step] = pos_curr[2]
        self.vel_x[self.step] = vel_curr[0]
        self.vel_y[self.step] = vel_curr[1]
        self.vel_z[self.step] = vel_curr[2]

        self.pos_x_d[self.step] = pos_d[0]
        self.pos_y_d[self.step] = pos_d[1]
        self.pos_z_d[self.step] = pos_d[2]
        self.vel_x_d[self.step] = vel_d[0]
        self.vel_y_d[self.step] = vel_d[1]
        self.vel_z_d[self.step] = vel_d[2]

        self.step += 1
    
    def reset(self):
        self.max_steps = self.max_timesteps // self.num_sims_per_env_step
        self.step = 0

        self.pos_x = np.zeros(self.max_steps)
        self.pos_y = np.zeros(self.max_steps)
        self.pos_z = np.zeros(self.max_steps)
        self.pos_x_d = np.zeros(self.max_steps)
        self.pos_y_d = np.zeros(self.max_steps)
        self.pos_z_d = np.zeros(self.max_steps)

        self.vel_x = np.zeros(self.max_steps)
        self.vel_y = np.zeros(self.max_steps)
        self.vel_z = np.zeros(self.max_steps)
        self.vel_x_d = np.zeros(self.max_steps)
        self.vel_y_d = np.zeros(self.max_steps)
        self.vel_z_d = np.zeros(self.max_steps)

    def plot_error(self):
        # Plotting
        fig, axs = plt.subplots(6, 1, figsize=(10, 18))
        timesteps = np.arange(self.max_steps)

        # pos_x and pos_x_d
        axs[0].plot(timesteps, self.pos_x, label='pos_x '+self.rec_obj, linestyle='-')
        axs[0].plot(timesteps, self.pos_x_d, label='pos_x_d '+self.rec_obj, linestyle='--')
        axs[0].set_title('x '+self.rec_obj)
        axs[0].legend()

        # pos_y and pos_y_d
        axs[1].plot(timesteps, self.pos_y, label='pos_y '+self.rec_obj, linestyle='-')
        axs[1].plot(timesteps, self.pos_y_d, label='pos_y_d '+self.rec_obj, linestyle='--')
        axs[1].set_title('y '+self.rec_obj)
        axs[1].legend()

        # pos_z and pos_z_d
        axs[2].plot(timesteps, self.pos_z, label='pos_z '+self.rec_obj, linestyle='-')
        axs[2].plot(timesteps, self.pos_z_d, label='pos_z_d '+self.rec_obj, linestyle='--')
        axs[2].set_title('z '+self.rec_obj)
        axs[2].legend()

        # vel_x and vel_x_d
        axs[3].plot(timesteps, self.vel_x, label='vel_x '+self.rec_obj, linestyle='-')
        axs[3].plot(timesteps, self.vel_x_d, label='vel_x_d '+self.rec_obj, linestyle='--')
        axs[3].set_title('vx '+self.rec_obj)
        axs[3].legend()

        # vel_y and vel_y_d
        axs[4].plot(timesteps, self.vel_y, label='vel_y '+self.rec_obj, linestyle='-')
        axs[4].plot(timesteps, self.vel_y_d, label='vel_y_d '+self.rec_obj, linestyle='--')
        axs[4].set_title('vy '+self.rec_obj)
        axs[4].legend()

        # vel_z and vel_z_d
        axs[5].plot(timesteps, self.vel_z, label='vel_z '+self.rec_obj, linestyle='-')
        axs[5].plot(timesteps, self.vel_z_d, label='vel_z_d '+self.rec_obj, linestyle='--')
        axs[5].set_title('vz '+self.rec_obj)
        axs[5].legend()

        # Layout adjustment
        plt.tight_layout()

        plt.show()

    def save_data(self):
        data = {
            'pos_x': self.pos_x,
            'pos_y': self.pos_y,
            'pos_z': self.pos_z,
            'vel_x': self.vel_x,
            'vel_y': self.vel_y,
            'vel_z': self.vel_z
        }

        # Create a pandas DataFrame from the dictionary
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        df.to_csv('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/state_data.csv', index=False)

        print("Data saved to state_data.csv")



if __name__ == "__main__":
    env = QuadrotorPayloadEnv()