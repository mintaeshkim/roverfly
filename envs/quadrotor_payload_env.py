# Helpers
import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
import numpy as np
from numpy import abs, asarray, clip, concatenate, copy, dot, exp, mean, ones, pi, round, sum, sqrt, tanh, zeros
from numpy.random import choice, uniform, normal
from numpy.linalg import inv, norm
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
from envs.utils.utility_functions import *
from envs.utils.rotation_transformations import *
from envs.utils.render_util import *
from envs.utils.mj_utils import *
from envs.utils.action_filter import ActionFilterButter
import time
import matplotlib.pyplot as plt


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 15.0}
  
class QuadrotorPayloadEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps:int = 4000,
        xml_file: str = "../assets/quadrotor_falcon_payload.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 1.0,
        env_num: int = 0,
        **kwargs
    ):
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)
        self.quadrotor_body_id = self.model.body(name="quadrotor").id
        self.payload_body_id = self.model.body(name="payload").id
        self.env_randomizer = EnvRandomizer(model=self.model)
        self.frame_skip = frame_skip
        self.control_scheme = "tvec"  # ["srt", "tvec", "ctbr"]
        np.random.seed(env_num)
        
        ##################################################
        #################### DYNAMICS ####################
        ##################################################
        # region
        self.policy_freq: float    = 100.0
        self.sim_freq: float       = 500.0 if self.control_scheme in ["ctbr", "tvec"] else self.policy_freq
        self.policy_dt: float      = 1 / self.policy_freq
        self.sim_dt: float         = 1 / self.sim_freq
        self.num_sims_per_env_step = int(self.sim_freq // self.policy_freq)
        # endregion
        ##################################################
        ###################### TIME ######################
        ##################################################
        # region
        self.max_timesteps: int  = max_timesteps
        self.timestep: int       = 0
        self.time_in_sec: float  = 0.0
        self.track_timesteps: int = 3000
        # endregion
        ##################################################
        #################### BOOLEANS ####################
        ##################################################
        # region
        self.is_io_history     = True
        self.is_delayed        = True
        self.is_env_randomized = True
        self.is_disturbance    = False
        self.is_full_traj      = False
        self.is_rotor_dynamics = True
        self.is_action_filter  = True
        self.is_ema_action     = False
        self.is_record_action  = True
        # endregion
        ##################################################
        ################## OBSERVATION ###################
        ##################################################
        # region
        self.env_num           = env_num
        self.s_dim             = 26
        self.d_dim             = 12
        self.a_dim             = 4 if self.control_scheme in ["srt", "ctbr"] else 3
        self.o_dim             = 312 if self.control_scheme in ["srt", "ctbr"] else 306
        self.history_len_short = 5
        self.history_len_long  = 10
        self.history_len       = self.history_len_short
        self.future_len        = 5
        self.s_buffer          = deque(zeros((self.history_len, self.s_dim)), maxlen=self.history_len)
        self.d_buffer          = deque(zeros((self.history_len, self.d_dim)), maxlen=self.history_len)
        self.a_buffer          = deque(zeros((self.history_len, self.a_dim)), maxlen=self.history_len)
        self.action_offset     = zeros(self.a_dim) if self.control_scheme in ["ctbr", "tvec"] else -0.4 * ones(4)
        self.force_offset      = 2.3 * ones(4)  # Warm start
        self.action_last       = self.action_offset
        self.num_episode       = 0
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
            "render_fps": int(self.policy_freq)
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
        # Quadrotor and payload characteristic values
        self.mQ = 0.835
        self.mP = 0.1
        self.g = 9.81
        self.JQ = np.array([[0.49, 0.00055, 0.002],
                            [0.00055, 0.53, 0.00054],
                            [0.002, 0.00054, 0.98]]) * 1e-2
        self.l = 0.1524
        self.d = self.l / sqrt(2)
        self.κ = 0.025
        self.A = inv(np.array([[1, 1, 1, 1],
                               [self.d, -self.d, -self.d, self.d],
                               [-self.d, -self.d, self.d, self.d],
                               [-self.κ, self.κ, -self.κ, self.κ]]))
        self.cable_length = 1.0
        
        # PID Control
        self.kPdφ, self.kPdθ, self.kPdψ = 1.0, 1.0, 0.8
        self.kIdφ, self.kIdθ, self.kIdψ = 0.0, 0.0, 0.0  # 0.0001, 0.0001, 0.00008
        self.kDdφ, self.kDdθ, self.kDdψ = 0.0, 0.0, 0.0  # 0.001, 0.001, 0.0008
        self.clipI = 0.15
        self.edφI, self.edθI, self.edψI = 0, 0, 0
        self.edφP_prev, self.edθP_prev, self.edψP_prev = 0, 0, 0

        # Additional PPID gains for thrust vector control (tvec)
        self.kp_att = np.array([8.0, 8.0, 3.0])  # Attitude position gains
        self.kp_rate = np.array([0.15, 0.15, 0.05])  # Rate proportional gains  
        self.ki_rate = np.array([0.0, 0.0, 0.0])  # Rate integral gains
        self.kd_rate = np.array([0.0001, 0.0001, 0.0])  # Rate derivative gains
        self.rate_integral = np.zeros(3)  # Rate error integral
        self.max_rate_integral = 0.15  # Anti-windup limit for rates
        self.moment_limit = [[-2,-2,-1],[2,2,1]]

        # Rotor Dynamics
        self.tau_up = 0.2164
        self.tau_down = 0.1644
        self.rotor_max_thrust = 7.5  # 14.981  # N
        self.max_thrust = 30  # N
        self.actual_forces = self.force_offset

        # Delay Parameters (From ground station to quadrotor)
        self.delay_range = [0.01, 0.03]  # 10 to 30 ms
        # To simulate the delay for data transmission
        self.action_queue_len = int(self.delay_range[1] / self.policy_dt) + 1
        self.delay_time = uniform(low=self.delay_range[0], high=self.delay_range[1])
        self.action_queue = deque([None] * self.action_queue_len, maxlen=self.action_queue_len)
        [self.action_queue.append([self.data.time, self.action_last]) for _ in range(self.action_queue_len)]
        # endregion

        # Disturbance Parameters
        self.disturbance_duration_range = [0, 0.25]  # Impulse
        self.force_disturbance_range = [-0.25, 0.25]  # N
        self.torque_disturbance_range = [-0.0025, 0.0025]  # N
        self.disturbance_duration = 0
        self.disturbance_start = 0

        self.time = 0

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def _set_action_space(self):
        lower_bound = np.full(self.a_dim, -1.0)
        upper_bound = np.full(self.a_dim, 1.0)
        self.action_space = Box(low=lower_bound, high=upper_bound, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self):
        if self.is_io_history: obs_shape = self.o_dim
        else: obs_shape = self.s_dim
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        return observation_space

    def _init_env(self):
        separator = "-" * 100
        env_info = (
            f"Environment {self.env_num} initialized.\n"
            f"Sample action: {self.action_space.sample()}\n"
            f"Action Space: {self.action_space}\n"
            f"Observation Space: {self.observation_space}\n"
            f"Time step (sec): {self.dt}\n"
        )
        print(env_info)
        print(separator)

    def _init_history_ff(self):
        s_curr = copy(self._get_state_curr())
        d_curr = concatenate([self.xQd[0] / self.pos_bound, self.xPd[0] / self.pos_bound, 
                              self.vQd[0] / self.vel_bound, self.vPd[0] / self.vel_bound])
        a_curr = copy(self.action)

        self.s_buffer.extend([s_curr] * self.history_len)
        self.d_buffer.extend([d_curr] * self.history_len)
        self.a_buffer.extend([a_curr] * self.history_len)
        if self.is_record_action: self.action_record = zeros((self.max_timesteps, self.a_dim))

    def reset(self, seed=None, randomize=None):
        # super().reset(seed=self.env_num)        
        if self.is_env_randomized:
            self.model, self.mP, self.cable_length, self.tau_up, self.tau_down = self.env_randomizer.randomize_env(self.model)
        self._reset_env()
        self._reset_model()
        self._reset_error()
        self._init_history_ff()
        self.info = self._get_reset_info()
        if self.is_action_filter:
            self.action_filter = ActionFilterButter(
                lowcut=[0],
                highcut=[5],
                sampling_rate=self.policy_freq,
                order=2,
                num_joints=3,
            )
        obs = self._get_obs()
        return obs, self.info
    
    def _reset_env(self):
        self.timestep     = 0
        self.time_in_sec  = 0.0
        self.action_last  = self.action_offset
        self.total_reward = 0
        self.terminated   = None
        self.info         = {}

    def _reset_model(self):
        if not self.is_full_traj: self.max_timesteps = self.track_timesteps

        self.traj = ut.CrazyTrajectoryPayload(
            tf=self.max_timesteps*self.policy_dt,
            ax=uniform(low=-2, high=2),
            ay=uniform(low=-2, high=2),
            az=uniform(low=-1, high=1),
            f1=uniform(low=-0.2, high=0.2),
            f2=uniform(low=-0.2, high=0.2),
            f3=uniform(low=-0.1, high=0.1)
        )
        # self.traj = ut.CrazyTrajectoryPayload(
        #     tf=self.max_timesteps*self.policy_dt,
        #     ax=0,
        #     ay=0,
        #     az=0,
        #     f1=0,
        #     f2=0,
        #     f3=0
        # )
        # self.type = np.random.choice(["crazy_1", "crazy_2", "crazy_3", "crazy_4",
        #                               "swing_1", "swing_2", "swing_3", "swing_4",
        #                               "circle_1", "circle_2", "circle_3", "circle_4",
        #                               "hover"])
        # self.traj = ut.PredefinedTrajectoryPayload(type="swing_1")
        
        if self.is_full_traj: self.traj = ut.FullCrazyTrajectoryPayload(tf=40, traj=self.traj)

        # self.traj.plot()
        # self.traj.plot3d_payload()

        """ Generate trajectory """
        self._generate_trajectory()

        """ Initial Perturbation """
        self._set_initial_state()

        """ Reset action """
        self.action = self.action_offset
        self.actual_forces = self.force_offset

        return self._get_obs()

    def _generate_trajectory(self):
        self.xPd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.vPd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.aPd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.daPd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.qd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.dqd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.d2qd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.xQd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.vQd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        for i in range(self.max_timesteps + self.history_len):
            self.xPd[i], self.vPd[i], self.aPd[i], self.daPd[i], self.qd[i], self.dqd[i], self.d2qd[i] = self.traj.get(i * self.policy_dt)
            self.xQd[i] = self.xPd[i] - self.qd[i] * self.cable_length
            self.vQd[i] = self.vPd[i] - self.dqd[i] * self.cable_length
        self.x_offset = self.pos_bound * np.array([uniform(-1, 1), uniform(-1, 1), 0 if self.is_full_traj else 2 * uniform(0.5, 1)])
        self.xPd += self.x_offset
        self.xQd += self.x_offset
        self.goal_pos = self.xPd[-1]
        self.dqd[0], self.d2qd[0] = zeros(3), zeros(3)

    def _set_initial_state(self):
        wx = 0.1 * uniform(size=3, low=-1, high=1)
        watt = (pi/12) * uniform(size=3, low=-1, high=1)
        wv = 0.1 * uniform(size=3, low=-1, high=1)
        wω = (pi/12) * uniform(size=3, low=-1, high=1)

        if self.is_full_traj:
            wx[-1], wv[-1] = 0, 0
            watt, wω = zeros(3), zeros(3)

        dPQ = self.cable_length - 0.01 * uniform(0, 1)
        psi = uniform(0, 2*pi)
        phi = (pi/6) * uniform(0, 1)
        
        xP = self.xPd[0] + wx
        xQ = xP + dPQ * np.array([cos(psi)*sin(phi), sin(psi)*sin(phi), cos(phi)])
        attP = self.init_qpos[10:14]
        attQ = euler2quat_raw(quat2euler_raw(self.init_qpos[3:7]) + watt)
        vP = self.vPd[0] + wv
        vQ = vP
        ωP = self.init_qvel[9:12]
        ωQ = self.init_qvel[3:6] + wω
        
        qpos = concatenate([xQ, attQ, xP, attP])
        qvel = concatenate([vQ, ωQ, vP, ωP])
        self.set_state(qpos, qvel)

    def _reset_error(self):
        self.edφI, self.edθI, self.edψI = 0, 0, 0
        self.edφP_prev, self.edθP_prev, self.edψP_prev = 0, 0, 0

    def _get_obs(self):
        self.obs_curr = self._get_obs_curr()  # 41

        s_buffer = asarray(self.s_buffer, dtype=np.float32).flatten()  # 130
        d_buffer = asarray(self.d_buffer, dtype=np.float32).flatten()  # 60
        a_buffer = asarray(self.a_buffer, dtype=np.float32).flatten()  # 15
        io_history = concatenate([s_buffer, d_buffer, a_buffer])  # 205

        xP_ff = self.xP - self.xPd[self.timestep : self.timestep + self.future_len]
        vP_ff = self.vP - self.vPd[self.timestep : self.timestep + self.future_len]
        xQd = self.xPd[self.timestep : self.timestep + self.future_len] - self.qd[self.timestep : self.timestep + self.future_len]
        vQd = self.vPd[self.timestep : self.timestep + self.future_len] - self.dqd[self.timestep : self.timestep + self.future_len]
        xQ_ff = self.xQ - xQd
        vQ_ff = self.vQ - vQd
        ff = concatenate([(xP_ff @ self.RQ).flatten(),
                          (vP_ff @ self.RQ).flatten(),
                          (xQ_ff @ self.RQ).flatten(),
                          (vQ_ff @ self.RQ).flatten()])  # 60

        return concatenate([self.obs_curr, io_history, ff])  # 306

    def _get_obs_curr(self):
        self.s_curr = self._get_state_curr()  # 26
        
        self.exP = self.xP - self.xPd[self.timestep]
        self.evP = self.vP - self.vPd[self.timestep]
        self.exQ = self.xQ - self.xQd[self.timestep]
        self.evQ = self.vQ - self.vQd[self.timestep]
        self.e_curr = concatenate([self.RQ.T @ self.exP, self.RQ.T @ self.evP, self.RQ.T @ self.exQ, self.RQ.T @ self.evQ])

        obs_curr = concatenate([self.s_curr, self.e_curr, self.action])  # 41

        return obs_curr

    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3] + clip(normal(loc=0, scale=0.01, size=3), -0.0025, 0.0025)
        self.RQ = euler2rot(quat2euler_raw(self.data.qpos[3:7])
                            + clip(normal(loc=0, scale=pi/60, size=3), -pi/120, pi/120))
        self.xP = self.data.qpos[7:10] + clip(normal(loc=0, scale=0.01, size=3), -0.0025, 0.0025)
        self.vQ = self.data.qvel[0:3] + clip(normal(loc=0, scale=0.02, size=3), -0.005, 0.005)
        self.ωQ = self.data.qvel[3:6] + clip(normal(loc=0, scale=pi/30, size=3), -pi/60, pi/60)
        self.vP = self.data.qvel[6:9] + clip(normal(loc=0, scale=0.02, size=3), -0.005, 0.005)

        return concatenate([self.xQ / self.pos_bound, self.RQ.flatten(), self.xP / self.pos_bound,
                            self.vQ / self.vel_bound, self.ωQ, self.vP / self.vel_bound, [self.mP, self.cable_length]])  # 26
   
    def step(self, action, restore=False):
        # 1. Simulate for Single Time Step
        self.raw_action = action
        self.action = self.action_filter.filter(self.raw_action) if self.is_action_filter else self.raw_action
        if self.is_ema_action: self.action = 0.2 * self.action + 0.8 * self.action_last
        if self.is_delayed: self.action_queue.append([self.data.time, self.action])
        if self.is_full_traj: self._apply_downwash()
        if self.is_disturbance: self._apply_disturbance()
        for _ in range(self.num_sims_per_env_step):
            action_apply = self._action_delay() if self.is_delayed else self.action
            self.do_simulation(action_apply, self.frame_skip)  # a_{t}
        if self.render_mode == "human": self.render()
        # 2. Get Observation
        obs = self._get_obs()
        # 3. Get Reward
        reward, reward_dict = self._get_reward()
        self.info["reward_dict"] = reward_dict
        # 4. Termination / Truncation
        terminated = self._terminated()
        truncated = self._truncated()
        self.info["terminated"] = terminated
        self.info["subreward"] = reward_dict
        # 5. Update Data
        self._update_data(reward=reward, step=True)

        return obs, reward, terminated, truncated, self.info
    
    def _action_delay(self):
        if self.data.time - self.action_queue[0][0] >= self.delay_time:
            action_delayed = self.action_queue.popleft()[1]
            self.delay_time = uniform(low=self.delay_range[0], high=self.delay_range[1])
        else:
            action_delayed = self.action_queue[0][1]
        return action_delayed

    def do_simulation(self, ctrl, n_frames):
        if self.control_scheme == "ctbr":
            ctrl = self._ctbr2srt(ctrl)
        elif self.control_scheme == "srt":
            ctrl = dual_tanh_srt(ctrl)
        else: # tvec
            ctrl = self._tvec2srt(ctrl)
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        if self.is_rotor_dynamics: self._rotor_dynamics(ctrl)
        else: self.actual_forces = ctrl
        self._apply_control(ctrl=self.actual_forces)
        mj.mj_step(self.model, self.data, nstep=n_frames)

    def _rotor_dynamics(self, ctrl):
        desired_forces = ctrl
        tau = np.array([self.tau_up if desired_forces[i] > self.actual_forces[i] else self.tau_down for i in range(4)])
        alpha = self.sim_dt / (tau + self.sim_dt)
        self.actual_forces = (1 - alpha) * self.actual_forces + alpha * desired_forces

    def _apply_control(self, ctrl):
        self.data.actuator("Motor0").ctrl[0] = ctrl[0]
        self.data.actuator("Motor1").ctrl[0] = ctrl[1]
        self.data.actuator("Motor2").ctrl[0] = ctrl[2]
        self.data.actuator("Motor3").ctrl[0] = ctrl[3]

    def _apply_downwash(self):
        if self.data.qpos[2] < 0.5:
            theta = uniform(0, 2 * np.pi)
            phi = uniform(0, np.pi / 2)
            downwash = np.array([sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)]) * (0.5 - self.data.qpos[2])
            self.data.xfrc_applied[self.quadrotor_body_id][:3] = downwash

    def _apply_disturbance(self):
        if self.data.time - self.disturbance_start < self.disturbance_duration:
            pass
        elif self.data.time - self.disturbance_start < self.disturbance_duration + 1:
            self.disturbance_wrench = np.zeros(6)
        else:
            self.disturbance_duration = uniform(low=self.disturbance_duration_range[0], high=self.disturbance_duration_range[1])
            force = uniform(low=self.force_disturbance_range[0], high=self.force_disturbance_range[1], size=3)    # Force  [N]
            torque = uniform(low=self.torque_disturbance_range[0], high=self.torque_disturbance_range[1], size=3) # Torque [Nm]
            self.disturbance_wrench = concatenate([force, torque])
            self.disturbance_start = self.data.time
        self.data.xfrc_applied[self.quadrotor_body_id][0:6] = self.disturbance_wrench

    def _tvec2srt(self, action):
        """Convert thrust vector command to single rotor thrusts using PPID control
           Control framework:
            High-level action → desired thrust vector & yaw
                        ↓
            Desired attitude (rotation matrix)
                        ↓
            P controller on attitude → cmd angular rate
                        ↓
            PID controller on angular rate → moments
                        ↓
            Thrust & moment → individual rotor forces
        """
        # Extract thrust vector components from action
        thrust_vec = concatenate([5 * tanh(action[:2]), [dual_tanh_tvec(action[2])]]) # [Fx, Fy, Fz] (N)
        yaw_sp = 0

        # Normalize e3_cmd (desired thrust direction)
        e3 = np.array([0, 0, 1])  # World z-axis
        e3_cmd = thrust_vec
        if np.isnan(norm(e3_cmd)) or norm(e3_cmd) < 1e-4: e3_cmd = e3
        e3_cmd = e3_cmd / norm(e3_cmd)

        # Compute desired rotation matrix
        e1_des = np.array([cos(yaw_sp), sin(yaw_sp), 0.0])
        e1_cmd = e1_des - dot(e1_des, e3_cmd) * e3_cmd
        e1_cmd = e1_cmd / norm(e1_cmd)
        e2_cmd = np.cross(e3_cmd, e1_cmd)
        R_des = np.column_stack((e1_cmd, e2_cmd, e3_cmd))
        
        # Convert to euler angles
        euler_des = rot2euler(R_des)
        
        # Attitude position control (P only)
        e_att = euler_des - rot2euler(self.RQ)
        
        # Compute desired angular rates (P from attitude)
        cmd_omega = self.kp_att * e_att
        
        # Rate control (PID)
        e_rate = cmd_omega - self.ωQ
        self.rate_integral = clip(self.rate_integral + e_rate * self.sim_dt, -self.max_rate_integral, self.max_rate_integral)
        rate_deriv = (e_rate - np.array([self.edφP_prev, self.edθP_prev, self.edψP_prev])) / self.sim_dt

        # Update previous errors for D term
        self.edφP_prev = e_rate[0]
        self.edθP_prev = e_rate[1]
        self.edψP_prev = e_rate[2]

        # Compute final moments using PPID structure
        M = (self.kp_rate * e_rate +  # P term on rates
             self.ki_rate * self.rate_integral +  # I term on rates
             self.kd_rate * rate_deriv)  # D term on rates
        M = clip(M,self.moment_limit[0], self.moment_limit[1])

        # Convert thrust and moments to rotor forces
        T = norm(thrust_vec)
        f = self.A @ np.array([T, M[0], M[1], M[2]])
        f = clip(f, 0, self.rotor_max_thrust)

        return f

    def _ctbr2srt(self, action):
        # action = self.action_last + 
        zcmd = self.max_thrust * (1 + action[0]) / 2  # dual_tanh_payload(action[0])
        dφd = action[1]  # tanh(action[1])
        dθd = action[2]  # tanh(action[2])
        dψd = action[3]  # tanh(action[3])

        self.edφP = dφd - self.ωQ[0]
        self.edφI = clip(self.edφI + self.edφP * self.sim_dt, -self.clipI, self.clipI)
        self.edφD = (self.edφP - self.edφP_prev) / self.sim_dt
        self.edφP_prev = self.edφP
        φcmd = clip(self.kPdφ * self.edφP + self.kIdφ * self.edφI + self.kDdφ * self.edφD, -2, 2)

        self.edθP = dθd - self.ωQ[1]
        self.edθI = clip(self.edθI + self.edθP * self.sim_dt, -self.clipI, self.clipI)
        self.edθD = (self.edθP - self.edθP_prev) / self.sim_dt
        self.edθP_prev = self.edθP
        θcmd = clip(self.kPdθ * self.edθP + self.kIdθ * self.edθI + self.kDdθ * self.edθD, -2, 2)

        self.edψP = dψd - self.ωQ[2]
        self.edψI = clip(self.edψI + self.edψP * self.sim_dt, -self.clipI, self.clipI)
        self.edψD = (self.edψP - self.edψP_prev) / self.sim_dt
        self.edψP_prev = self.edψP
        ψcmd = clip(self.kPdψ * self.edψP + self.kIdψ * self.edψI + self.kDdψ * self.edψD, -2, 2)

        Mcmd = np.array([φcmd, θcmd, ψcmd])
        f = self.A @ concatenate([[zcmd], Mcmd]).reshape((4,1))
        f = clip(f.flatten(), 0, self.rotor_max_thrust)

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
        # Past
        self.s_buffer.append(self.s_curr)
        self.d_buffer.append(concatenate([self.xQd[self.timestep] / self.pos_bound, self.xPd[self.timestep] / self.pos_bound,
                                          self.vQd[self.timestep] / self.vel_bound, self.vPd[self.timestep] / self.vel_bound]))
        self.a_buffer.append(self.raw_action)
        self.action_last = self.action
        if self.is_record_action and self.timestep < self.max_timesteps: self.action_record[self.timestep] = self.action
        # Present
        self.time_in_sec = round(self.time_in_sec + self.policy_dt, 2)
        self.timestep += 1
        self.total_reward += reward

    def _get_reward(self):
        weights = {'xP':1.0, 'vP':0.0, 'xQ':0.0, 'vQ':0.0,
                   'ψQ':0.25, 'ωQ':0.25, 'a':0.25, 'Δa':0.25, 'dq':0.1}
        # weights = {'xP':1.0, 'vP':0.25, 'xQ':0.1, 'vQ':0.025,
        #            'ψQ':0.25, 'ωQ':0.25, 'a':0.25, 'Δa':0.25, 'dq':0.1}
        weights = {k: v / sum(list(weights.values())) for k, v in weights.items()}
        scales = {'xP':1.0/0.1, 'vP':1.0/0.4, 'xQ':1.0/0.1, 'vQ':1.0/0.4,
                  'ψQ':1.0/(pi/4), 'ωQ':1.0/0.25, 'a':1.0/0.5, 'Δa':1.0/0.1, 'dq':1.0/0.4}
        errors = {
            'xP': norm(self.data.qpos[7:10] - self.xPd[self.timestep]),
            'vP': norm(self.data.qvel[6:9] - self.vPd[self.timestep]),
            'xQ': norm(self.data.qpos[0:3] - self.xQd[self.timestep]),
            'vQ': norm(self.data.qvel[0:3] - self.vQd[self.timestep]),
            'ψQ': abs(quat2euler_raw(self.data.qpos[3:7])[2]),
            'ωQ': norm(self.data.qvel[3:6]),
            'a': norm(self.raw_action),
            'Δa': sum(exp(-np.linspace(0,1,self.history_len)) * 
                      [norm(a-b) for a,b in zip(list(self.a_buffer)[1:]+[self.action], list(self.a_buffer)+[self.action])]) 
                  / sum(exp(-np.linspace(0,1,self.history_len))),
            'dq': norm((self.data.qvel[0:3] - self.data.qvel[6:9]) / self.cable_length if not self.cable_length == 0 else 0)
        }
        rewards = {k: np.exp(-scales[k]*errors[k]) for k in weights}
        weighted_rewards = {k: weights[k]*rewards[k] for k in weights}
        total_reward = (weighted_rewards['xP'] + weighted_rewards['ψQ']) * (1 + weighted_rewards['ωQ'] + weighted_rewards['dq']) + weighted_rewards['a'] + weighted_rewards['Δa']
        return total_reward, rewards

    def _terminated(self):
        xQ = self.data.qpos[0:3]
        xP = self.data.qpos[7:10]
        vP = self.data.qvel[6:9]
        attQ = quat2euler_raw(self.data.qpos[3:7])

        xPd = self.xPd[self.timestep] if self.timestep < self.max_timesteps else self.goal_pos
        vPd = self.vPd[self.timestep] if self.timestep < self.max_timesteps else zeros(3)
        exP = norm(xP - xPd)
        evP = norm(vP - vPd)

        if 5.0 <= self.time_in_sec <= 35.0 and xQ[2] < 0:  # Crash except for takeoff and landing
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Crashed | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            if self.is_record_action: self.plot_action()
            return True
        if exP > self.pos_err_bound:
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Pos error: {pos_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  pos_err=round(exP, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            if self.is_record_action: self.plot_action()
            return True
        elif evP > self.vel_err_bound:
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Vel error: {vel_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  vel_err=round(evP, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            if self.is_record_action: self.plot_action()
            return True
        elif not(abs(attQ) < pi/2).all():
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Att error: {att} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  att=round(attQ, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            if self.is_record_action: self.plot_action()
            return True
        elif self.timestep >= self.max_timesteps:
            """ TEST """
            # self.test_record_Q.plot_error()
            # self.test_record_Q.reset()
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Max time: {time} | Final pos error: {pos_err} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  time=round(self.time_in_sec, 2),
                  pos_err=round(exP, 2),
                  rew=round(self.total_reward, 1)))
            if self.is_record_action: self.plot_action()
            return True
        else:
            return False

    def _truncated(self):
        return False

    def render(self):
        if self.mujoco_renderer.viewer is not None:
            if self.timestep == 0 or self.timestep == 1: setup_viewer(self.mujoco_renderer.viewer)
        #     self.mujoco_renderer.viewer.scn.ngeom = 0
        #     self.render_trajectory(self.mujoco_renderer.viewer.scn, self.xPd)
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def plot_action(self):
        fig, axs = plt.subplots(self.a_dim, 1, figsize=(12, 10))
        timesteps = np.arange(self.max_timesteps)

        axs[0].plot(timesteps, self.action_record[:, 0], label='action_0', linestyle='-')
        axs[0].set_title('action_0')
        axs[0].set_ylim([-1, 1])
        axs[0].legend()

        axs[1].plot(timesteps, self.action_record[:, 1], label='action_1', linestyle='-')
        axs[1].set_title('action_1')
        axs[1].set_ylim([-1, 1])
        axs[1].legend()

        axs[2].plot(timesteps, self.action_record[:, 2], label='action_2', linestyle='-')
        axs[2].set_title('action_2')
        axs[2].set_ylim([-1, 1])
        axs[2].legend()

        if self.control_scheme in ["ctbr", "srt"]:
            axs[3].plot(timesteps, self.action_record[:, 3], label='action_3', linestyle='-')
            axs[3].set_title('action_3')
            axs[3].set_ylim([-1, 1])
            axs[3].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = QuadrotorPayloadEnv()