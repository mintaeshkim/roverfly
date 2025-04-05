# Helpers
import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
import numpy as np
from numpy import abs, asarray, clip, concatenate, copy, dot, exp, mean, pi, round, sum, sqrt, tanh, zeros
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
from envs.utils.action_filter import ContinuousActionFilter
import time
import matplotlib.pyplot as plt


DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0, "distance": 15.0}
  
class QuadrotorEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(
        self,
        max_timesteps:int = 4000,
        xml_file: str = "../assets/quadrotor_falcon.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 1.0,
        env_num: int = 0,
        **kwargs
    ):
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)
        self.quadrotor_body_id = self.model.body(name="quadrotor").id
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
        self.max_timesteps: int   = max_timesteps
        self.timestep: int        = 0
        self.time_in_sec: float   = 0.0
        self.track_timesteps: int = 3000
        # endregion
        ##################################################
        #################### BOOLEANS ####################
        ##################################################
        # region
        self.is_io_history     = True
        self.is_delayed        = True
        self.is_env_randomized = True
        self.is_disturbance    = True
        self.is_full_traj      = False
        self.is_rotor_dynamics = False
        self.is_action_filter  = False
        self.is_record_action  = True
        # endregion
        ##################################################
        ################## OBSERVATION ###################
        ##################################################
        # region
        self.env_num           = env_num
        self.s_dim             = 18
        self.a_dim             = 4
        self.o_dim             = 198
        self.history_len_short = 5
        self.history_len_long  = 10
        self.history_len       = self.history_len_short
        self.future_len        = 5
        self.s_buffer          = deque(zeros((self.history_len, self.s_dim)), maxlen=self.history_len)  # [x, R, v, ω]
        self.d_buffer          = deque(zeros((self.history_len, 6)), maxlen=self.history_len)  # [xQd, vQd]
        self.a_buffer          = deque(zeros((self.history_len, self.a_dim)), maxlen=self.history_len)
        self.action_offset     = zeros(4) if self.control_scheme in ["ctbr", "tvec"] else -0.46 * np.ones(4)
        self.force_offset      = 2.0 * np.ones(4)  # Warm start (for rotor dynamics)
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
        # region
        self._init_env()
        # endregion
        ##################################################
        ############### LOW-LEVEL CONTROL ################
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
        
        # PID Control (ctbr)
        self.kPdφ, self.kPdθ, self.kPdψ = 1.0, 1.0, 0.8
        self.kIdφ, self.kIdθ, self.kIdψ = 0.0, 0.0, 0.0
        self.kDdφ, self.kDdθ, self.kDdψ = 0.0, 0.0, 0.0
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
        self.disturbance_duration_range = [0, 0.5]  # Impulse
        self.force_disturbance_range = [-0.5, 0.5]  # N
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
        d_curr = concatenate([self.xQd[0] / self.pos_bound, self.vQd[0] / self.vel_bound])
        a_curr = copy(self.action)

        self.s_buffer.extend([s_curr] * self.history_len)
        self.d_buffer.extend([d_curr] * self.history_len)
        self.a_buffer.extend([a_curr] * self.history_len)
        if self.is_record_action: self.action_record = zeros((self.max_timesteps, self.a_dim))

    def reset(self, seed=None, randomize=None):
        # super().reset(seed=self.env_num)
        self._reset_env()
        self._reset_model()
        self._reset_error()
        self._init_history_ff()
        obs = self._get_obs()
        self.info = self._get_reset_info()
        if self.is_env_randomized: self.model = self.env_randomizer.randomize_env(self.model)
        if self.is_action_filter:
            self.action_filter = ContinuousActionFilter(
                history_len=self.history_len,
                a_dim=self.a_dim,
                lipschitz_const = 10.0,
                a_buffer=self.a_buffer,
                dt=self.policy_dt
            )
        return obs, self.info
  
    def _reset_env(self):
        self.timestep     = 0
        self.time_in_sec  = 0.0
        self.action_last  = self.action_offset
        self.total_reward = 0
        self.terminated   = None
        self.info         = {}

        """ TEST """
        # self.test_record_Q = TestRecord(max_timesteps=self.max_timesteps,
        #                                 record_object='Q',
        #                                 num_sims_per_env_step=self.num_sims_per_env_step)

    def _reset_model(self):
        if not self.is_full_traj: self.max_timesteps = self.track_timesteps
        
        self.traj = ut.CrazyTrajectory(
            tf=self.track_timesteps*self.policy_dt,
            ax=choice([-1,1])*2.0,
            ay=choice([-1,1])*2.0,
            az=choice([-1,1])*1.0,
            f1=choice([-1,1])*0.2,
            f2=choice([-1,1])*0.2,
            f3=choice([-1,1])*0.1
        )

        # self.traj = ut.CrazyTrajectory(tf=self.track_timesteps*self.policy_dt, ax=1, ay=-1, az=0.5, f1=0.2, f2=0.3, f3=0.25)

        if self.is_full_traj: self.traj = ut.FullCrazyTrajectory(tf=40, traj=self.traj)

        # self.traj.plot()
        # self.traj.plot3d()

        """ Generate trajectory """
        self._generate_trajectory()

        """ Initial perturbation """
        self._set_initial_state()

        """ Reset action """
        self.action = self.action_offset
        self.actual_forces = self.force_offset

        return self._get_obs()

    def _generate_trajectory(self):
        self.xQd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.vQd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        self.aQd = zeros((self.max_timesteps + self.history_len, 3), dtype=np.float32)
        for i in range(self.max_timesteps + self.history_len):
            self.xQd[i], self.vQd[i], self.aQd[i] = self.traj.get(i * self.policy_dt)
        self.x_offset = self.pos_bound * np.array([uniform(-1, 1), uniform(-1, 1), 0 if self.is_full_traj else 2 * uniform(0.5, 1)])
        self.xQd += self.x_offset
        self.goal_pos = self.xQd[-1]

    def _set_initial_state(self):
        wx = 0.1 * uniform(size=3, low=-1, high=1)
        watt = (pi/12) * uniform(size=3, low=-1, high=1)
        wv = 0.1 * uniform(size=3, low=-1, high=1)
        wω = (pi/12) * uniform(size=3, low=-1, high=1)
        
        if self.is_full_traj:
            wx[-1], wv[-1] = 0, 0
            watt, wω = zeros(3), zeros(3)

        xQ = self.xQd[0] + wx
        attQ = euler2quat_raw(quat2euler_raw(self.init_qpos[3:7]) + watt)
        vQ = self.vQd[0] + wv
        ωQ = self.init_qvel[3:6] + wω
        
        qpos = concatenate([xQ, attQ])
        qvel = concatenate([vQ, ωQ])
        self.set_state(qpos, qvel)

    def _reset_error(self):
        self.edφI, self.edθI, self.edψI = 0, 0, 0
        self.edφP_prev, self.edθP_prev, self.edψP_prev = 0, 0, 0

    def _get_obs(self):
        self.obs_curr = self._get_obs_curr()

        s_buffer = asarray(self.s_buffer, dtype=np.float32).flatten()
        d_buffer = asarray(self.d_buffer, dtype=np.float32).flatten()
        a_buffer = asarray(self.a_buffer, dtype=np.float32).flatten()
        io_history = concatenate([s_buffer, d_buffer, a_buffer])

        xQ_ff = self.xQ - self.xQd[self.timestep : self.timestep + self.future_len]
        vQ_ff = self.vQ - self.vQd[self.timestep : self.timestep + self.future_len]
        ff = concatenate([(xQ_ff @ self.R).flatten(), (vQ_ff @ self.R).flatten()])

        # print(round(self.xQ, 3))
        # print(round(self.xQd[self.timestep], 3))
        # print()

        return concatenate([self.obs_curr, io_history, ff])

    def _get_obs_curr(self):
        self.s_curr = self._get_state_curr()
        
        self.exQ = self.xQ - self.xQd[self.timestep]
        self.evQ = self.vQ - self.vQd[self.timestep]
        self.e_curr = concatenate([self.R.T @ self.exQ, self.R.T @ self.evQ])

        obs_curr = concatenate([self.s_curr, self.e_curr, self.action])  # 28

        return obs_curr
    
    def _get_state_curr(self):
        self.xQ = self.data.qpos[0:3] + clip(normal(loc=0, scale=0.01, size=3), -0.0025, 0.0025)
        self.R = euler2rot(quat2euler_raw(self.data.qpos[3:7])
                           + clip(normal(loc=0, scale=pi/60, size=3), -pi/120, pi/120))
        self.vQ = self.data.qvel[0:3] + clip(normal(loc=0, scale=0.02, size=3), -0.005, 0.005)
        self.ω = self.data.qvel[3:6] + clip(normal(loc=0, scale=pi/30, size=3), -pi/60, pi/60)
        return concatenate([self.xQ / self.pos_bound, self.R.flatten(), self.vQ / self.vel_bound, self.ω])

    def step(self, action, restore=False):
        # 1. Simulate for Single Time Step
        self.action = self.action_filter.filter(action) if self.is_action_filter else action
        if self.is_delayed: self.action_queue.append([self.data.time, self.action])
        if self.is_full_traj: self._apply_downwash()
        if self.is_disturbance: self._apply_disturbance()
        for _ in range(self.num_sims_per_env_step):
            self.action = self._action_delay() if self.is_delayed else self.action
            self.do_simulation(self.action, self.frame_skip)  # a_{t}
        if self.render_mode == "human": self.render()
        # 2. Get Observation
        obs = self._get_obs()  # s_{t+1}
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
            ctrl = self.rotor_max_thrust * (ctrl + 1) / 2
        else: # tvec
            ctrl = self._tvec2srt(ctrl)
        self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl, n_frames):
        self.actual_forces = self._rotor_dynamics(ctrl) if self.is_rotor_dynamics else ctrl
        self._apply_control(ctrl=ctrl)  # if rotor dynamics: ctrl=self.actual_forces
        mj.mj_step(self.model, self.data, nstep=n_frames)

    def _rotor_dynamics(self, ctrl):
        desired_forces = ctrl
        tau = np.array([self.tau_up if desired_forces[i] > self.actual_forces[i] else self.tau_down for i in range(4)])
        alpha = self.sim_dt / (tau + self.sim_dt)
        return (1 - alpha) * self.actual_forces + alpha * desired_forces

    def _apply_control(self, ctrl):
        self.data.actuator("Motor0").ctrl[0] = ctrl[0]  # data.ctrl[1] # front
        self.data.actuator("Motor1").ctrl[0] = ctrl[1]  # data.ctrl[2] # back
        self.data.actuator("Motor2").ctrl[0] = ctrl[2]  # data.ctrl[3] # left
        self.data.actuator("Motor3").ctrl[0] = ctrl[3]  # data.ctrl[4] # right

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
        yaw_sp = 0; self.action[3] = 0   # action[3] * np.pi    # Desired yaw angle (just set to 0 for now)

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
        euler_curr = quat2euler_raw(self.data.qpos[3:7])
        omega_curr = self.data.qvel[3:6]
        
        # Attitude position control (P only)
        e_att = euler_des - euler_curr
        
        # Compute desired angular rates (P from attitude)
        cmd_omega = self.kp_att * e_att
        
        # Rate control (PID)
        e_rate = cmd_omega - omega_curr
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
        zcmd = dual_tanh(action[0])
        dφd = tanh(action[1])
        dθd = tanh(action[2])
        dψd = tanh(action[3])

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

    def _update_data(self, reward, step=False):
        """ TEST """
        # self._record()
        # Past
        self.s_buffer.append(self.s_curr)
        self.d_buffer.append(concatenate([self.xQd[self.timestep] / self.pos_bound, self.vQd[self.timestep] / self.vel_bound]))
        self.a_buffer.append(self.action)
        self.action_last = self.action
        if self.is_record_action and self.timestep < self.max_timesteps: self.action_record[self.timestep] = self.action
        # Present
        self.time_in_sec = round(self.time_in_sec + self.policy_dt, 2)
        self.timestep += 1
        self.total_reward += reward
    
    def _record(self):
        self.test_record_Q.record(pos_curr=self.xQ, vel_curr=self.vQ, pos_d=self.xQd[self.timestep], vel_d=self.vQd[self.timestep])

    def _get_reward(self):
        names = ['xQ_rew', 'vQ_rew', 'ψQ_rew', 'ωQ_rew', 'Δa_rew']
        
        w_xQ = 1.0
        w_vQ = 0.25  # 0.1
        w_ψQ = 0.25  # 0.5
        w_ωQ = 0.25  # 0.25
        w_Δa = 0.25  # 0.25

        reward_weights = np.array([w_xQ, w_vQ, w_ψQ, w_ωQ, w_Δa])
        weights = reward_weights / sum(reward_weights)

        scale_xQ = 1.0/0.1
        scale_vQ = 1.0/0.4
        scale_ψQ = 1.0/(pi/4)
        scale_ωQ = 1.0/1.0
        scale_Δa = 1.0/0.1  # 1.0/0.2

        exQ = norm(self.data.qpos[0:3] - self.xQd[self.timestep], ord=2)
        evQ = norm(self.data.qvel[0:3] - self.vQd[self.timestep], ord=2)
        eψQ = abs(quat2euler_raw(self.data.qpos[3:7])[2])
        eωQ = norm(self.data.qvel[3:6], ord=2)
        # eΔa = norm(self.action - self.action_last, ord=2)
        action_seq = list(self.a_buffer) + [self.action]
        weights = exp(-np.linspace(0, 1, self.history_len))
        diffs = [norm(action_seq[i] - action_seq[i - 1], ord=2)
                 for i in range(1, self.history_len + 1)]
        eΔa = np.sum(weights * diffs) / np.sum(weights)

        rewards = exp(-np.array([scale_xQ, scale_vQ, scale_ψQ, scale_ωQ, scale_Δa])
                      *np.array([exQ, evQ, eψQ, eωQ, eΔa]))
        reward_dict = dict(zip(names, weights * rewards))
        total_reward = sum(weights * rewards)

        return total_reward, reward_dict

    def _terminated(self):
        xQ = self.data.qpos[0:3]
        vQ = self.data.qvel[0:3]
        attQ = quat2euler_raw(self.data.qpos[3:7])

        xQd = self.xQd[self.timestep] if self.timestep < self.max_timesteps else self.goal_pos
        vQd = self.vQd[self.timestep] if self.timestep < self.max_timesteps else np.zeros(3)
        exQ = norm(xQ - xQd)
        evQ = norm(vQ - vQd)

        if 5.0 <= self.time_in_sec <= 35.0 and xQ[2] < 0.01:  # Crash except for takeoff and landing
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Crashed | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            return True
        if exQ > self.pos_err_bound:
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Pos error: {pos_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  pos_err=round(exQ, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            return True
        elif evQ > self.vel_err_bound:
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Vel error: {vel_err} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  vel_err=round(evQ, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
            return True
        elif not(abs(attQ) < pi/2).all():
            self.num_episode += 1
            print("Env {env_num} | Ep {epi} | Att error: {att} | Time: {time} | Reward: {rew}".format(
                  env_num=self.env_num,
                  epi=self.num_episode,
                  att=round(attQ, 2),
                  time=round(self.time_in_sec, 2),
                  rew=round(self.total_reward, 1)))
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
                  pos_err=round(exQ, 2),
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
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def plot_action(self):
        fig, axs = plt.subplots(4, 1, figsize=(12, 10))
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

        axs[3].plot(timesteps, self.action_record[:, 3], label='action_3', linestyle='-')
        axs[3].set_title('action_3')
        axs[3].set_ylim([-1, 1])
        axs[3].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = QuadrotorEnv()
    env.reset()
    print(env._get_obs())