# Helpers
import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
import numpy as np
from numpy import abs, clip, concatenate, copy, exp, mean, pi, round, sum, sqrt
from numpy.random import choice, uniform, normal
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as R
from typing import Dict, Union
from collections import deque
# Gym
from gymnasium.spaces import Box
from gymnasium import utils
# Mujoco
import mujoco as mj
from mujoco_gym.mujoco_env import MujocoEnv
# ETC
import envs.utils.utility_trajectory as ut
from envs.utils.env_randomizer import EnvRandomizer
from envs.utils.geo_tools import hat, vee
from envs.utils.utility_functions import *
from envs.utils.rotation_transformations import *
import time
import matplotlib.pyplot as plt


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 10.0}
  
class QuadrotorEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps = 10000,  # 20 seconds
        xml_file: str = "../assets/quadrotor_falcon.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 1.0,
        env_num: int = 0,
        **kwargs
    ):
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)
        self.body_id = self.model.body(name="quadrotor").id
        self.env_randomizer = EnvRandomizer(model=self.model)
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
        self.is_action_bound    = False
        self.is_io_history      = True
        self.is_delayed         = True
        # endregion
        ##################################################
        ################## OBSERVATION ###################
        ##################################################
        # region
        self.env_num           = env_num
        self.s_dim             = 18
        self.a_dim             = 4
        self.history_len_short = 5
        self.history_len_long  = 10
        self.history_len       = self.history_len_short
        self.future_len        = 5
        self.delay_len         = 3
        self.o_dim             = 198
        self.s_buffer          = deque(np.zeros((self.history_len, self.s_dim)), maxlen=self.history_len)  # [x, R, v, ω]
        self.d_buffer          = deque(np.zeros((self.history_len, 6)), maxlen=self.history_len)  # [xQd, vQd]
        self.a_buffer          = deque(np.zeros((self.history_len, self.a_dim)), maxlen=self.history_len)
        self.action_last       = np.array([-1, 0, 0, 0])
        self.num_episode       = 0
        self.history_epi       = {'setpoint': deque([0]*10, maxlen=10),
                                  'curve': deque([0]*10, maxlen=10)}
        self.progress          = {'setpoint': 1e-3,
                                  'curve': 1e-3}
        self.action_space      = self._set_action_space()
        self.observation_space = self._set_observation_space()
        # endregion
        ##################################################
        ##################### BOUNDS #####################
        ##################################################
        # region
        self.pos_bound = 3.0
        self.vel_bound = 5.0
        self.pos_err_bound = 0.5
        self.vel_err_bound = 2.0
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
            "render_fps": int(round(1.0 / self.sim_dt))
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
        self._init_env()
        # endregion
        ##################################################
        ################### PD CONTROL ###################
        ##################################################
        # region
        # Quadrotor Characteristic Values
        self.mQ = 0.8
        self.mP = 0.1
        self.g = 9.81
        self.JQ = np.array([[0.49, 0, 0],
                            [0, 0.53, 0],
                            [0, 0, 0.98]]) * 1e-2
        self.l = 0.1524
        self.d = self.l / sqrt(2)
        self.κ = 0.025
        self.A = inv(np.array([[1, 1, 1, 1],
                                         [self.d, -self.d, -self.d, self.d],
                                         [-self.d, -self.d, self.d, self.d],
                                         [-self.κ, self.κ, -self.κ, self.κ]]))
        
        # PID Control
        self.kPdφ, self.kPdθ, self.kPdψ = 1.0, 1.0, 0.8
        self.kIdφ, self.kIdθ, self.kIdψ = 0.0, 0.0, 0.0  # 0.0001, 0.0001, 0.00008
        self.kDdφ, self.kDdθ, self.kDdψ = 0.0, 0.0, 0.0  # 0.001, 0.001, 0.0008
        self.clipI = 0.15
        self.edφI, self.edθI, self.edψI = 0, 0, 0
        self.edφP_prev, self.edθP_prev, self.edψP_prev = 0, 0, 0

        # Rotor Dynamics
        self.tau_up = 0.2164
        self.tau_down = 0.1644
        self.rotor_max_thrust = 7.5  # 14.981  # N
        self.max_thrust = 30  # N
        self.actual_forces = (self.mQ * self.g / 4) * np.ones(4)

        # Delay Parameters (From ground station to quadrotor)
        self.delay_range = [0.01, 0.02]  # 10 to 20 ms
        # To simulate the delay for data transmission
        self.action_queue = deque([[self.data.time, self.action_last]], maxlen=int(self.delay_range[1] / self.policy_dt))
        # endregion

        self.time = 0

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip
  
    def _set_action_space(self):
        # CTBR
        low = np.array([-1, -1, -1, -1])
        high = np.array([1, 1, 1, 1])
        if self.is_action_bound: self.action_space = Box(low=0.2*np.ones(4), high=0.8*np.ones(4))
        else: self.action_space = Box(low=low, high=high)
        return self.action_space

    def _set_observation_space(self):
        if self.is_io_history: obs_shape = self.o_dim
        else: obs_shape = self.s_dim
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        return observation_space

    def _init_env(self):
        print("Environment {} created".format(self.env_num))
        print("Sample action: {}".format(self.action_space.sample()))
        print("Action Space: {}".format(np.array(self.action_space)))
        print("Observation Space: {}".format(np.array(self.observation_space)))
        print("Time step(sec): {}".format(self.dt))
        # print("Policy frequency(Hz): {}".format(self.policy_freq))
        # print("Num sims / Env step: {}".format(self.num_sims_per_env_step))
        print("-"*100)

    def _init_history_ff(self):
        s_curr = copy(self._get_state_curr())
        d_curr = concatenate([self.xQd[0] / self.pos_bound, self.vQd[0] / self.vel_bound])
        a_curr = copy(self.action)
        [self.s_buffer.append(s_curr) for _ in range(self.history_len)]
        [self.d_buffer.append(d_curr) for _ in range(self.history_len)]
        [self.a_buffer.append(a_curr) for _ in range(self.history_len)]

    def reset(self, seed=None, randomize=None):
        # super().reset(seed=self.env_num)
        self._reset_env()
        self._reset_model()
        self._reset_error()
        self._init_history_ff()
        obs = self._get_obs()
        self.info = self._get_reset_info()
        self.model = self.env_randomizer.randomize_env(self.model)
        return obs, self.info
  
    def _reset_env(self):
        self.timestep     = 0  # discrete timestep, k
        self.time_in_sec  = 0.0  # time
        
        self.action_last  = np.array([-1, 0, 0, 0])
        self.q_last       = np.array([0,0,-1])
        
        self.total_reward = 0
        self.terminated   = None
        self.info         = {}

        """ TEST """
        # self.test_record_Q = TestRecord(max_timesteps=self.max_timesteps,
        #                                 record_object='Q',
        #                                 num_sims_per_env_step=self.num_sims_per_env_step)

    def _reset_model(self):
        """ Compute progress """
        self.progress['setpoint'] = (mean(self.history_epi['setpoint']) / self.max_timesteps)
        self.progress['curve'] = (mean(self.history_epi['curve']) / self.max_timesteps)
        
        """ TEST """
        # self.progress['setpoint'] = 1.0
        # self.progress['curve'] = 0.35

        """ Choose task """
        if self.progress['setpoint'] < 0.5:
            self.stage = 1
            self.traj_type = choice(['setpoint', 'curve'], p=[0.9, 0.1])
        else:
            self.stage = 2
            self.traj_type = choice(['setpoint', 'curve'], p=[0.1, 0.9])

        """ TEST """
        # self.stage = 2
        self.traj_type = 'setpoint'

        """ Set trajectory parameters """
        if self.traj_type == 'setpoint':
            self.traj = ut.CrazyTrajectory(tf=self.max_timesteps*self.policy_dt, ax=0, ay=0, az=0, f1=0, f2=0, f3=0)
            self.difficulty = self.stage * self.progress["setpoint"]
        if self.traj_type == 'curve':
            self.traj = ut.CrazyTrajectory(tf=self.max_timesteps*self.policy_dt,
                                           ax=choice([-1,1])*3*self.progress["curve"],
                                           ay=choice([-1,1])*3*self.progress["curve"],
                                           az=choice([-1,1])*3*self.progress["curve"],
                                           f1=choice([-1,1])*0.5*self.progress["curve"],
                                           f2=choice([-1,1])*0.5*self.progress["curve"],
                                           f3=choice([-1,1])*0.5*self.progress["curve"])
            self.difficulty = self.stage * self.progress["curve"]
        
        # self.traj.plot()
        # self.traj.plot3d_payload()

        """ Compute trajectory """
        self.xQd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.vQd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.aQd = np.zeros((self.max_timesteps + self.history_len, 3))
        for i in range(self.max_timesteps + self.history_len):
            self.xQd[i], self.vQd[i], self.aQd[i] = self.traj.get(i*self.policy_dt)
        self.x_offset = self.pos_bound * uniform(size=3, low=-1, high=1)
        self.xQd += self.x_offset
        self.goal_pos = self.xQd[-1]

        """ Initial Perturbation """
        # self.perturbation = 1e-5
        self.perturbation = self.progress["curve"]

        wx = 0.05 * uniform(0, self.perturbation)
        watt = (pi/36) * uniform(0, self.perturbation)
        wv = 0.1 * uniform(0, self.perturbation)
        wω = (pi/18) * uniform(0, self.perturbation)
        
        xQ = self.xQd[0] + uniform(size=3, low=-wx, high=wx)
        attQ = euler2quat_raw(quat2euler_raw(self.init_qpos[3:7]) + uniform(size=3, low=-watt, high=watt))
        vQ = self.vQd[0] + uniform(size=3, low=-wv, high=wv)
        ωQ = self.init_qvel[3:6] + uniform(size=3, low=-wω, high=wω)
        
        qpos = concatenate([xQ, attQ])
        qvel = concatenate([vQ, ωQ])
        self.set_state(qpos, qvel)

        self.action = np.array([-1, 0, 0, 0])
        self.actual_forces = (self.mQ * self.g / 4) * np.ones(4)

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
        self.obs_curr = self._get_obs_curr()  # 28

        # Past
        s_buffer = np.array(self.s_buffer, dtype=object).flatten()  # 90
        d_buffer = np.array(self.d_buffer, dtype=object).flatten()  # 30
        a_buffer = np.array(self.a_buffer, dtype=object).flatten()  # 20
        io_history = concatenate([s_buffer, d_buffer, a_buffer])  # 140

        # Future
        xQ_ff = self.xQ - self.xQd[self.timestep : self.timestep + self.future_len]
        vQ_ff = self.vQ - self.vQd[self.timestep : self.timestep + self.future_len]
        ff = concatenate([(xQ_ff @ self.R).flatten(), (vQ_ff @ self.R).flatten()])  # 30

        obs_full = concatenate([self.obs_curr, io_history, ff])

        return obs_full

    def _get_obs_curr(self):
        self.s_curr = self._get_state_curr()
        
        self.exQ = self.xQ - self.xQd[self.timestep]
        self.evQ = self.vQ - self.vQd[self.timestep]
        self.e_curr = concatenate([self.R.T @ self.exQ, self.R.T @ self.evQ])

        obs_curr = concatenate([self.s_curr, self.e_curr, self.action])  # 28

        return obs_curr
    
    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3] + clip(normal(loc=0, scale=0.01, size=3), -0.01, 0.01)
        self.R = euler2rot(quat2euler_raw(self.data.qpos[3:7])
                           + clip(normal(loc=0, scale=pi/36, size=3), -pi/60, pi/60))
        self.vQ = self.data.qvel[0:3] + clip(normal(loc=0, scale=0.02, size=3), -0.02, 0.02)
        self.ω = self.data.qvel[3:6] + clip(normal(loc=0, scale=pi/18, size=3), -pi/30, pi/30)
        return concatenate([self.xQ / self.pos_bound, self.R.flatten(), self.vQ / self.vel_bound, self.ω])

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
        qd = - A / norm(A)

        return qd, dqd

    def step(self, action, restore=False):
        # 1. Simulate for Single Time Step
        self.action = action
        if self.is_delayed: self.action_queue.append([self.data.time, self.action])
        for _ in range(self.num_sims_per_env_step):
            self.do_simulation(self.action, self.frame_skip)  # a_{t}
        if self.render_mode == "human": self.render()
        # 2. Get Observation
        obs_full = self._get_obs()  # s_{t+1}
        # 3. Get Reward
        reward, reward_dict = self._get_reward()
        self.info["reward_dict"] = reward_dict
        # 4. Termination / Truncation
        terminated = self._terminated()
        truncated = self._truncated()
        # 5. Update Data
        self._update_data(reward=reward, step=True)

        return obs_full, reward, terminated, truncated, {}
    
    def do_simulation(self, ctrl, n_frames):
        if self.is_delayed:
            delay_time = uniform(self.delay_range[0], self.delay_range[1])
            if self.data.time - self.action_queue[0][0] >= delay_time:
                ctrl = self.action_queue.popleft()[1]
        ctrl = self._ctbr2srt(ctrl)
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self._rotor_dynamics(ctrl)
        self._apply_control(ctrl=self.actual_forces)
        mj.mj_step(self.model, self.data, nstep=n_frames)

    def _rotor_dynamics(self, ctrl):
        desired_forces = ctrl
        tau = np.zeros(4)
        for i in range(4):
            tau[i] = self.tau_up if desired_forces[i] > self.actual_forces[i] else self.tau_down
        alpha = self.sim_dt / (tau + self.sim_dt)
        self.actual_forces = (1 - alpha) * self.actual_forces + alpha * desired_forces

    def _apply_control(self, ctrl):
        self.data.actuator("Motor0").ctrl[0] = ctrl[0]  # data.ctrl[1] # front
        self.data.actuator("Motor1").ctrl[0] = ctrl[1]  # data.ctrl[2] # back
        self.data.actuator("Motor2").ctrl[0] = ctrl[2]  # data.ctrl[3] # left
        self.data.actuator("Motor3").ctrl[0] = ctrl[3]  # data.ctrl[4] # right

    def _ctbr2srt(self, action):
        zcmd = self.max_thrust * (action[0] + 1) / 2
        dφd = action[1] / 2
        dθd = action[2] / 2
        dψd = action[3] / 2

        self.edφP = dφd - self.ω[0]
        self.edφI = clip(self.edφI + self.edφP * self.sim_dt, -self.clipI, self.clipI)
        self.edφD = (self.edφP - self.edφP_prev) / self.sim_dt
        self.edφP_prev = self.edφP
        φcmd = clip(self.kPdφ * self.edφP + self.kIdφ * self.edφI + self.kDdφ * self.edφD, -2, 2)

        self.edθP = dθd - self.ω[1]
        self.edθI = clip(self.edθI + self.edθP * self.sim_dt, -self.clipI, self.clipI)
        self.edθD = (self.edθP - self.edθP_prev) / self.sim_dt
        self.edθP_prev = self.edθP
        θcmd = clip(self.kPdθ * self.edθP + self.kIdθ * self.edθI + self.kDdθ * self.edθD, -2, 2)

        self.edψP = dψd - self.ω[2]
        self.edψI = clip(self.edψI + self.edψP * self.sim_dt, -self.clipI, self.clipI)
        self.edψD = (self.edψP - self.edψP_prev) / self.sim_dt
        self.edψP_prev = self.edψP
        ψcmd = clip(self.kPdψ * self.edψP + self.kIdψ * self.edψI + self.kDdψ * self.edψD, -2, 2)

        # Original
        Mcmd = np.array([φcmd, θcmd, ψcmd])
        f = self.A @ concatenate([[zcmd], Mcmd]).reshape((4,1))
        f = clip(f.flatten(), 0, self.rotor_max_thrust)

        # region
        # print("f: ", round(f, 3))
        # print("zcmd: ", round(zcmd, 3))
        # print("φcmd: ", round(φcmd, 3))
        # print("θcmd: ", round(θcmd, 3))
        # print("ψcmd: ", round(ψcmd, 3))
        # print("P: ", round(self.kPdφ * self.edφP, 3))
        # print("I: ", round(self.kIdφ * self.edφI, 3))
        # print("D: ", round(self.kDdφ * self.edφD, 3))
        # endregion

        return f

    def _update_data(self, reward, step=True):
        """ TEST """
        # self._record()
        # Past
        self.s_buffer.append(self.s_curr)
        self.d_buffer.append(concatenate([self.xQd[self.timestep] / self.pos_bound, self.vQd[self.timestep] / self.vel_bound]))
        self.a_buffer.append(self.action)
        self.action_last = self.action
        # Present
        self.time_in_sec = round(self.time_in_sec + self.policy_dt, 2)
        self.timestep += self.num_sims_per_env_step
        self.total_reward += reward
    
    def _record(self):
        self.test_record_Q.record(pos_curr=self.xQ, vel_curr=self.vQ, pos_d=self.xQd[self.timestep], vel_d=self.vQd[self.timestep])

    def _get_reward(self):
        names = ['xQ_rew', 'vQ_rew', 'ψQ_rew', 'ωQ_rew']
        
        w_xQ = 1.0
        w_vQ = 0.5
        w_ψQ = 1.0
        w_ωQ = 0.5
        w_a  = 0.5

        reward_weights = np.array([w_xQ, w_vQ, w_ψQ, w_ωQ, w_a])
        weights = reward_weights / sum(reward_weights)

        scale_xQ = 1.0/0.5
        scale_vQ = 1.0/2.0
        scale_ψQ = 1.0/(pi/2)
        scale_ωQ = 1.0/0.25
        scale_a  = 1.0/4.0

        ψQd = 0
        ψQ  = quat2euler_raw(self.data.qpos[3:7])[2]

        exQ = norm(self.exQ, ord=2)
        evQ = norm(self.evQ, ord=2)
        eψQ = abs(ψQ - ψQd)
        eωQ = norm(self.ω, ord=2)
        ea = norm(np.array([0.2, 1, 1, 1]) * self.action, ord=2)

        rewards = exp(-np.array([scale_xQ, scale_vQ, scale_ψQ, scale_ωQ, scale_a])
                      *np.array([exQ, evQ, eψQ, eωQ, ea]))
        reward_dict = dict(zip(names, weights * rewards))
        
        if self.traj_type == "setpoint":
            total_reward = sum(weights * rewards)
        else:
            total_reward = sum(weights * rewards) * (1 + 0.5 * self.difficulty)

        return total_reward, reward_dict

    def _terminated(self):
        xQ = self.data.qpos[0:3]
        vQ = self.data.qvel[0:3]
        attQ = quat2euler_raw(self.data.qpos[3:7])

        xQd = self.xQd[self.timestep] if self.timestep < self.max_timesteps else self.goal_pos
        vQd = self.vQd[self.timestep] if self.timestep < self.max_timesteps else np.zeros(3)
        exQ = norm(xQ - xQd)
        evQ = norm(vQ - vQd)

        if exQ > self.pos_err_bound:
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Pos error: {pos_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(round(self.difficulty, 2)),
                  pos_err=round(exQ, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            return True
        elif evQ > self.vel_err_bound:
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Vel error: {vel_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(round(self.difficulty, 2)),
                  vel_err=round(evQ, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            return True
        elif not(abs(attQ) < pi/2).all():
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Att error: {att} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(round(self.difficulty, 2)),
                  att=round(attQ, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            return True
        elif self.timestep >= self.max_timesteps:
            """ TEST """
            # self.test_record_Q.plot_error()
            # self.test_record_Q.reset()
            self.num_episode += 1
            self.history_epi[self.traj_type].append(self.timestep)
            print("Env {env_num} | Ep {epi} | St {stage} | Traj: {traj_type} | Max time: {time} | Final pos error: {pos_err} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  stage=self.stage,
                  traj_type=self.traj_type + " " + str(round(self.difficulty, 2)),
                  time=round(self.time_in_sec, 2),
                  pos_err=round(exQ, 2),
                  rew=round(self.total_reward, 1)))
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



if __name__ == "__main__":
    env = QuadrotorEnv()
    env.reset()
    print(env._get_obs())