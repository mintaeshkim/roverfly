#!/home/hrg/miniconda3/envs/RLpy38ros/bin/python

import os, sys
import time

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append('/home/hrg/Documents/qrotor_ws/src/qrotor_ros/qrotor_ground/scripts/rl/envs')

import onnx
import onnxruntime as ort

import envs.utility_trajectory as ut
from envs.geo_tools import *
from envs.utility_functions import *
from envs.rotation_transformations import *
from envs.action_filter import ActionFilterButter

import numpy as np
import copy
from collections import deque

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Vector3
from qrotor_firmware.msg import RLCommand

from dynamic_reconfigure.server import Server
from qrotor_ground.cfg import RLOffboardManagerConfig

vehicle_name = 'white_falcon'
onnx_path = "/home/hrg/Documents/qrotor_ws/src/qrotor_ros/qrotor_ground/scripts/rl/train/best_model_10172024.onnx"


class PIDController:
    def __init__(self, kp, ki, kd, initial_setpoint, dt, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = initial_setpoint
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.previous_error = 0.0
        # self.last_time = None
        self.dt = dt

    def reset(self, initial_setpoint):
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = None
        
    def compute(self, process_variable, new_setpoint=None):
        # Update setpoint if a new one is provided
        if new_setpoint is not None:
            self.setpoint = new_setpoint

        error = self.setpoint - process_variable
        
        # Calculate time difference
        # if self.last_time is None: dt = 0
        # else: dt = current_time - self.last_time
        # self.last_time = current_time
        
        # Proportional term
        p_term = self.kp * error
        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral
        # Derivative term
        if self.dt > 0: d_term = self.kd * (error - self.previous_error) / self.dt
        else: d_term = 0
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits if specified
        if self.output_limits is not None:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Store error for next iteration
        self.previous_error = error
        
        return output



class RLOffboardNode(object):
    def __init__(self):

        # Check Model
        self.onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(self.onnx_model)
        self.ort_sess = ort.InferenceSession(onnx_path)
        print('RL_model_loaded')

        # subscriber to odometry
        self._quad_odom_sub = rospy.Subscriber('/'+vehicle_name+'/odometry/mocap', Odometry, self._quad_odom_cb, queue_size=1)
        self._pyld_odom_sub = rospy.Subscriber('/'+vehicle_name+'/payload/odometry/mocap', Odometry, self._pyld_odom_cb, queue_size=1)
        self._quad_odom = Odometry()
        self._pyld_odom = Odometry()

        self._pub_actions = rospy.Publisher('/'+vehicle_name+'/actions', RLCommand, queue_size=1)

        self._waiting_for_payload_odom = True
        self._waiting_for_quadrotor_odom = True

        # Initialize flags 
        self._send = False
        # self._enable_control = False
        self._enable_RL = False
        self._takeoff = False
        self._setpoint_mode = False
        self._land = False
        self._falcon_states = {"ground":0, "takeoff":1, "land":2, "air":3}
        self._falcon_state = self._falcon_states["ground"]
        self._time_record = 0

        # Variables
        self._home_position = np.array([0., 0., 0.])
        self._home_takeoff_position = np.array([0., 0., 0.])
        self.setpoint = [0,0,1.5]
        
        # Time Variables
        self.sim_freq    = 500.0  # Hz
        self.policy_freq = 100.0  # Hz
        self.sim_dt      = 1 / self.sim_freq
        self.policy_dt   = 1 / self.policy_freq
        self.num_sims_per_env_step = int(self.sim_freq // self.policy_freq)
        self.prev_time_ = rospy.get_time() - self.policy_dt
        self.curr_time_ = rospy.get_time()
        
        # Initialize PID controller (Drone Position Control)
        k_xy = np.array([1.8, 0.05, 50])/2 # Kp, Ki, Kd 
        k_z = np.array([8, 0.25, 2])  # Kp, Ki, Kd 
        self.x_vel_Ctrl = PIDController(k_xy[0], k_xy[1], k_xy[2], self.policy_dt, 0, [-np.pi/6,np.pi/6])  # limit pitch angle -np.pi/6,np.pi/6
        self.y_vel_Ctrl = PIDController(k_xy[0], k_xy[1], k_xy[2], self.policy_dt, 0, [-np.pi/6,np.pi/6])  # limit roll angle
        self.z_vel_Ctrl = PIDController(k_z[0], k_z[1], k_z[2], self.policy_dt, 0, [-16, 8]) # limit list thrust

        # Trajectory Parameters
        self.max_timesteps:int = 10 * int(self.sim_freq)  # 10 Seconds
        self.timestep:int = 0

        # Observation Variables
        self.history_len_short = 5
        self.history_len_long  = 10
        self.history_len       = self.history_len_short
        self.future_len        = 3
        
        self.s_len             = 18  # if payload: xQ, ΘQ, xP, vQ, ΩQ, vP
        self.e_buffer          = deque(np.zeros((self.history_len, 6)), maxlen=self.history_len)
        self.a_buffer          = deque(np.zeros((self.history_len_long, 4)), maxlen=self.history_len)
        self.s_buffer          = deque(np.zeros((self.history_len_long, self.s_len)), maxlen=self.history_len_long)

        self.action = np.zeros(4)
        self.q_last = np.array([0,0,-1])

        self.mQ = 0.8
        self.mP = 0.1
        #-------------------------------
        self.get_traj()

        self._dynreconfig_server = Server(RLOffboardManagerConfig, self._dynreconfig_server_callback)


        # start the node
        # self.get_obs()  # Sanity check
        self.run()
    
    def _dynreconfig_server_callback(self, config, level):
        print('------------')
        if config['send']:
            self._send = True
            print("send")
        else:
            print('not send')
            self._send = False
            config['enable_RL'] = False
            config['setpoint'] = False

        if config['enable_control']:
            self._enable_control =  True
            print("Enable Control")
        else:
            self._enable_control = False
            config['enable_RL'] = False
            self._falcon_state = self._falcon_states["ground"]

        if config['enable_RL']:
            if not self._send:
                print("Not send yet. Cannot start RL")
            # elif not self._falcon_state == self._falcon_states["air"]:
            #     print('Cannot Start RL trajectory tracking. Not in air mode')
            else:
                print("RL trajetory tracking!")
                config['setpoint'] = False
                self._enable_RL = True
        else:
            config['enable_RL'] = False
            self._enable_RL = False
            config['setpoint'] = True

        if config['setpoint']:
            print('Setpoint Mode')
            self._setpoint_mode = True
        else:
            config['setpoint'] = False
            self._setpoint_mode = False

        if config['takeoff']:
            if not self._send:
                print("Starting sending before takeoff")
            elif not self._falcon_state == self._falcon_states["ground"]:
                print('Cannot takeoff. Not in ground mode')
            elif not self._enable_control:
                print("Cannot takeoff. Need to enable control first!")
            else:
                print('takeoff=',config['takeoff'])
                self.request_takeoff()
                config['enable_control'] = True
                self._falcon_state = self._falcon_states["air"]
                print("falcon status:",self._falcon_states["air"])
            config['takeoff'] = False
                
        if config['land']:
            if not self._send:
                print("Starting sending before land")
            elif not self._falcon_state == self._falcon_states["air"]:
                print('Cannot land. Not in air mode')
            else:
                print('land=',config['land'])
                self.request_landing()
                self._falcon_state = self._falcon_states["ground"]
            config['land'] = False

        if config['reset']:
            self.timestep = 0
            config['reset'] = False

    
        return config
    
    def request_setpoint(self):
        self._setpoint_mode = True
        self._takeoff = False
        self._land = False

    def request_takeoff(self):
        self._home_position = copy.deepcopy(self.xQ)
        self._home_takeoff_position = self._home_position + np.array([0., 0., 1.5])
        self.setpoint = self._home_takeoff_position

        self._takeoff = False
        self._setpoint_mode = True
        self._enable_RL = False
        # rospy.Timer(rospy.Duration(1),self.request_setpoint())
       
    def request_landing(self):
        setpoint = copy.deepcopy(self.xQ)
        setpoint[2] = self._home_position[2]
        self.setpoint = setpoint

        self._land = True
        self._setpoint_mode = True
        self._enable_RL = False
        # rospy.Timer(rospy.Duration(1),self.landed())

    def landed(self):
        self._land = False
        self._enable_control = False

        # --------------------------------- #
    
    def _init_history_ff(self):
        [self.e_buffer.append(self.e_curr) for _ in range(self.history_len)]
        [self.a_buffer.append(self.action) for _ in range(self.history_len)]

    def update_time(self, Enable_RL=False):
        self.curr_time_ = rospy.get_time()
        self.dt_        = self.curr_time_ - self.prev_time_
        self.prev_time_ = self.curr_time_
        rospy.loginfo('Frequency: %f Hz',1/self.dt_)

    def _quad_odom_cb(self, msg):
        if self._waiting_for_quadrotor_odom:
            self._waiting_for_quadrotor_odom = False
        self._quad_odom = copy.deepcopy(msg)
        # print('quad_odom_cb')
        return

    def _pyld_odom_cb(self, msg):
        if self._waiting_for_payload_odom:
            self._waiting_for_payload_odom = False
        self._pyld_odom = copy.deepcopy(msg)
        # print('pyld_odom_cb')
        return

    """ Quadrotor + Payload """
    """
    def get_traj(self):
        print("Generateing Trajectory....")
        self.traj_type = "setpoint"
        self.traj = ut.CrazyTrajectoryPayload(tf=self.max_timesteps*self.policy_dt, ax=0, ay=0, az=0, f1=0, f2=0, f3=0)
        # self.traj = ut.SmoothTraj5Payload(np.array([0, 0, 0.5]), np.array([0, 0, 1]), 5)
        self.xPd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.vPd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.aPd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.daPd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.qd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.dqd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.d2qd = np.zeros((self.max_timesteps + self.history_len, 3))
        for i in range(self.max_timesteps + self.history_len):
            self.xPd[i], self.vPd[i], self.aPd[i], self.daPd[i], self.qd[i], self.dqd[i], self.d2qd[i] = self.traj.get(i * self.policy_dt)
        self.goal_pos = self.xPd[-1]
        self.dqd[0], self.d2qd[0] = np.zeros(3), np.zeros(3)
        print(self.xPd[:10,:])
        print('Done')

    def get_obs(self):
        self.obs_curr = self.get_obs_curr()

        e_buffer = np.array(self.e_buffer, dtype=object).flatten()
        a_buffer = np.array(self.a_buffer, dtype=object)[-self.history_len:,:].flatten()
        s_buffer = np.array(self.s_buffer, dtype=object)[-self.history_len:,:]
        Θ_buffer = np.array([s_buffer[i,3:6] for i in range(self.history_len)]).flatten()  # 15
        Ω_buffer = s_buffer[:,12:15].flatten()  # 15
        io_history = np.concatenate([e_buffer, a_buffer, Θ_buffer, Ω_buffer])

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
    
    def get_obs_curr(self):
        self.xP = np.array([self._pyld_odom.pose.pose.position.x, self._pyld_odom.pose.pose.position.y, self._pyld_odom.pose.pose.position.z])
        self.vP = np.array([self._pyld_odom.twist.twist.linear.x, self._pyld_odom.twist.twist.linear.y, self._pyld_odom.twist.twist.linear.z])
        self.xQ = np.array([self._quad_odom.pose.pose.position.x, self._quad_odom.pose.pose.position.y, self._quad_odom.pose.pose.position.z])
        self.vQ = np.array([self._quad_odom.twist.twist.linear.x, self._quad_odom.twist.twist.linear.y, self._quad_odom.twist.twist.linear.z])
        self.ΩQ = np.array([self._quad_odom.twist.twist.angular.x, self._quad_odom.twist.twist.angular.y, self._quad_odom.twist.twist.angular.z])
        self.qQ = np.array([self._quad_odom.pose.pose.orientation.w, self._quad_odom.pose.pose.orientation.x, self._quad_odom.pose.pose.orientation.y, self._quad_odom.pose.pose.orientation.z])
        self.R = quat2rot(self.qQ)
        self.ΘQ = quat2euler_raw(self.qQ)
        # return np.concatenate((xQ, vQ, quatQ, OmegaQ, xL, vL))  # this quat is [x,y,z,w] 
        
        self.exP = self.xP - self.xPd[self.timestep]
        self.evP = self.vP - self.vPd[self.timestep]

        self.q = (self.xP - self.xQ) / np.linalg.norm((self.xP - self.xQ))
        self.dq = (self.q - self.q_last) / self.policy_dt
        self.q_last = self.q
        qd, dqd = self._get_qd()

        self.xQd = self.xPd[self.timestep] - qd
        self.vQd = self.vPd[self.timestep] - dqd

        self.exQ = self.xQ - self.xQd
        self.evQ = self.vQ - self.vQd

        self.e_curr = np.concatenate([self.R.T @ self.exP, self.R.T @ self.evP, self.R.T @ self.exQ, self.R.T @ self.evQ])

        obs_curr = np.concatenate([self.e_curr,
                                   self.R.T @ self.q, self.R.T @ self.dq, self.R.T @ qd, self.R.T @ dqd,
                                   self.action, self.R.flatten(), self.ΩQ])
        
        return obs_curr
    
    def _get_qd(self):
        l = 1.0
        kx = 2 * np.diag([0.5, 0.5, 0.5])
        kv = 2 * np.diag([0.75, 0.75, 0.75])
        aPd = self.aPd[self.timestep] if self.timestep < self.max_timesteps else np.zeros(3)
        dqd = self.dqd[self.timestep] if self.timestep < self.max_timesteps else self.dqd[-1]

        Fff = (self.mQ + self.mP) * (aPd + np.array([0,0,9.81])) + self.mQ * l * np.dot(self.dq, self.dq) * self.q
        Fpd = -kx @ self.exP - kv @ self.evP
        A = Fff + Fpd
        qd = - A / np.linalg.norm(A)

        return qd, dqd

    def _update_data(self, step=True):
        if step:
            s_curr = np.concatenate([self.xQ, self.ΘQ, self.xP, self.vQ, self.ΩQ, self.vP])
            self.s_buffer.append(s_curr)
            self.e_buffer.append(np.copy(self.e_curr))
            self.a_buffer.append(np.copy(self.action))
            
            self.a_last = self.action
            self.timestep += self.num_sims_per_env_step
    """

    def get_traj(self):
        print("Generateing Trajectory....")
        self.traj_type = "setpoint"
        self.traj = ut.CrazyTrajectory(tf=self.max_timesteps*self.policy_dt, ax=0, ay=0, az=0, f1=0, f2=0, f3=0)
        self.xQd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.vQd = np.zeros((self.max_timesteps + self.history_len, 3))
        self.aQd = np.zeros((self.max_timesteps + self.history_len, 3))
        for i in range(self.max_timesteps + self.history_len):
            self.xQd[i], self.vQd[i], self.aQd[i] = self.traj.get(i*self.policy_dt)
        self.goal_pos = self.xQd[-1]
        print(self.xQd[:10,:])
        print('Done')

    def get_obs(self):
        # Present
        self.obs_curr = self._get_obs_curr()

        # Past
        e_buffer = np.array(self.e_buffer, dtype=object).flatten()
        a_buffer = np.array(self.a_buffer, dtype=object).flatten()
        io_history = np.concatenate([e_buffer, a_buffer])  # 50

        # Future
        xQ_ff = self.xQ - self.xQd[self.timestep : self.timestep + self.future_len]
        vQ_ff = self.vQ - self.vQd[self.timestep : self.timestep + self.future_len]
        ff = np.concatenate([(xQ_ff @ self.R).flatten(), (vQ_ff @ self.R).flatten()])  # 18

        obs_full = np.concatenate([self.obs_curr, io_history, ff])

        return obs_full

    def get_obs_curr(self):
        self.xQ = np.array([self._quad_odom.pose.pose.position.x, self._quad_odom.pose.pose.position.y, self._quad_odom.pose.pose.position.z])
        self.vQ = np.array([self._quad_odom.twist.twist.linear.x, self._quad_odom.twist.twist.linear.y, self._quad_odom.twist.twist.linear.z])
        
        self.exQ = self.xQ - self.xQd[self.timestep]
        self.evQ = self.vQ - self.vQd[self.timestep]
        
        self.q = np.array([self._quad_odom.pose.pose.orientation.w, self._quad_odom.pose.pose.orientation.x, self._quad_odom.pose.pose.orientation.y, self._quad_odom.pose.pose.orientation.z])
        self.ω = np.array([self._quad_odom.twist.twist.angular.x, self._quad_odom.twist.twist.angular.y, self._quad_odom.twist.twist.angular.z])
        self.R = quat2rot(self.q)

        self.e_curr = np.concatenate([self.R.T @ self.exQ, self.R.T @ self.evQ])

        obs_curr = np.concatenate([self.e_curr,                             # 6                                                    
                                   self.R.flatten(), self.ω, self.action])  # 16

        return obs_curr

    def _update_data(self, reward, step=True):
        if step:
            self.e_buffer.append(self.e_curr)
            self.a_buffer.append(self.action)
            self.action_last = self.action
            # Present
            self.time_in_sec = np.round(self.time_in_sec + self.policy_dt, 2)
            self.timestep += self.num_sims_per_env_step
    
    def run(self):
        """
        Ground Station Node Run
        """
        print('--------------------------')
        gnd_loop_rate = rospy.Rate(self.policy_freq)
        i = 0.0
        # running ros loop
        while (not rospy.is_shutdown()):
            
            if self._enable_control:
                # print("Mode:", self._falcon_state)
                
                self.update_time(self._enable_RL)
                if not self._waiting_for_quadrotor_odom: # and not self._waiting_for_payload_odom:
                    # print('not waiting')

                    observation = self.get_obs().reshape((1,90)).astype(np.float32)
                    if self._falcon_state == self._falcon_states["ground"]:
                        self.setpoint = self.xQ
                    
                    if self._enable_RL: # and self._falcon_state == self._falcon_states["air"]:  # Trajectory
                        rospy.logwarn_once("RL Trajectory Tracking")
                        action, _, _ = self.ort_sess.run(None, {"input": observation})
                        # print(action)
                        self.action = action.flatten()
                        self._update_data(step=True)

                        if self.timestep == self.max_timesteps:
                            self._enable_RL = False
                            self._timestep = 0
                            self._setpoint_mode = True
                    
                    else:  # Position PID
                        rospy.logwarn_once('PID')
                        # print("PID Drone Control")
                        # self.get_obs()
                        euler = quat2euler_raw(self.q)
                        K_xy = 0.7
                        K_roll_pitch = 2.5
                        K_yaw = 1
                        K_z = 1
                        
                        pitch_d = self.x_vel_Ctrl.compute(self.vQ[0], new_setpoint=np.clip(K_xy*(self.setpoint[0]-self.xQ[0]),-2,2))  # Limit desired x velocity
                        roll_d = -self.y_vel_Ctrl.compute(self.vQ[1], new_setpoint=np.clip(K_xy*(self.setpoint[1]-self.xQ[1]),-2,2))  # Limit desired y velocity
                        # pitch_d = self.x_vel_Ctrl.compute(self.xQ[0], new_setpoint=self.setpoint[0])  # Limit desired x velocity
                        # roll_d = -self.y_vel_Ctrl.compute(self.xQ[1], new_setpoint=self.setpoint[1])  # Limit desired y velocity
                        # pitch_d = 0
                        # roll_d = 0
                        yaw_d = 0 

                        ang_rate_d = np.array([np.clip (K_roll_pitch * (roll_d - euler[0]), -2*np.pi, 2*np.pi),
                                            np.clip (K_roll_pitch * (pitch_d - euler[1]), -2*np.pi, 2*np.pi),
                                            np.clip (K_yaw * (yaw_d - euler[2]), -2*np.pi, 2*np.pi)])/np.pi
                        thrust_d = self.z_vel_Ctrl.compute(self.vQ[2], new_setpoint=np.clip(K_z*(self.setpoint[2]-self.xQ[2]), -5, 5))/8   # Limit desired z velocity
                        
                        self.action = [thrust_d, ang_rate_d[0], ang_rate_d[1], ang_rate_d[2]]

                        # print(self.xQ, euler, self.action)
                        
                    # print(self.action)

                else:
                    rospy.logwarn_once('waiting for odom')
            
            else:
                self.action = [-0.5, 0, 0 ,0] # Ground
            
                # Sending out action commands #
            if self._send:
                msg = RLCommand()
                msg.thrust = 10 * (action[0] + 1) / 2 # Thrust (normalized) multiply mg to get thrust in N
                i = i + 0.001
                msg.angular_velocity = Vector3(self.action[1], self.action[2], self.action[3]) # ang vel (rad/s)
                self._pub_actions.publish(msg)
                # print("sent command msg")
                # print("sent action:",self.action)
                current_time = time.time()
                milliseconds = int((current_time - int(current_time)) * 1000)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}.{milliseconds:03d}", " action ", i)

            # loop
            gnd_loop_rate.sleep()
        return

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    print("----------------------------------")
    print(node_name)    
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)

    RLOffboardNode()
    
    rospy.loginfo('Shutting down [%s] node' % node_name)