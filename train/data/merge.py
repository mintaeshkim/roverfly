import numpy as np

# Load and merge s_record_setpoint
s_record_setpoint_0 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_0.0.csv', delimiter=',')[1:]
s_record_setpoint_1 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_0.1.csv', delimiter=',')[1:]
s_record_setpoint_2 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_0.2.csv', delimiter=',')[1:]
s_record_setpoint_3 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_0.3.csv', delimiter=',')[1:]
s_record_setpoint_4 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_0.41.csv', delimiter=',')[1:]
s_record_setpoint_5 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_1.01.csv', delimiter=',')[1:]
s_record_setpoint_6 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_1.21.csv', delimiter=',')[1:]
s_record_setpoint_7 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_1.42.csv', delimiter=',')[1:]
s_record_setpoint = np.concatenate([s_record_setpoint_0, s_record_setpoint_1, s_record_setpoint_2, s_record_setpoint_3, s_record_setpoint_4, s_record_setpoint_5, s_record_setpoint_6, s_record_setpoint_7])
np.savetxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint.csv', s_record_setpoint, delimiter=',')

# Load and merge a_record_setpoint
a_record_setpoint_0 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_0.0.csv', delimiter=',')[1:]
a_record_setpoint_1 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_0.1.csv', delimiter=',')[1:]
a_record_setpoint_2 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_0.2.csv', delimiter=',')[1:]
a_record_setpoint_3 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_0.3.csv', delimiter=',')[1:]
a_record_setpoint_4 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_0.41.csv', delimiter=',')[1:]
a_record_setpoint_5 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_1.01.csv', delimiter=',')[1:]
a_record_setpoint_6 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_1.21.csv', delimiter=',')[1:]
a_record_setpoint_7 = np.loadtxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_1.42.csv', delimiter=',')[1:]
a_record_setpoint = np.concatenate([a_record_setpoint_0, a_record_setpoint_1, a_record_setpoint_2, a_record_setpoint_3, a_record_setpoint_4, a_record_setpoint_5, a_record_setpoint_6, a_record_setpoint_7])
np.savetxt('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint.csv', a_record_setpoint, delimiter=',')