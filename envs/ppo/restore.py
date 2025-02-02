import sys
sys.path.append('..')
from stable_baselines3.common import tf_util as U
from Flappy_Integrated.flappy.envs.train.run import train
import tensorflow as tf
from DigitEnv.DigitEnv import DigitEnv
import time
from utility.utility import *
import argparse


class Writecsv():
    def __init__(self,filename):
        self.motor_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        motor_header = 'left_hip_roll,left_hip_yaw,left_hip_pitch,left_knee,left_toe,' + \
                       'right_hip_roll,right_hip_yaw,right_hip_pitch,right_knee,right_toe\n'
        pol_header = 'time,vx,vy,z,rx,ry,rz,' + motor_header
        ref_header = 'time,vx,vy,z,' + motor_header
        self.f = open("../data/results/" + filename, 'w')
        self.fref = open("../data/results/" + filename.split('.csv')[0]+'_ref.csv', 'w')
        self.f.write(pol_header)
        self.fref.write(ref_header)

    def write(self, time, qpos, qvel, ref_joints):
        qpos_str = list(map(str, qpos))
        qvel_str = list(map(str, qvel))
        ref_joints_str = list(map(str, ref_joints))
        self.f.write(str(time) +
                     ''.join([',' + qvel_str[i] for i in range(2)]) +
                     ',' + qpos_str[2] +
                     ''.join(',' + str(Quat2Rxyz(qpos[3:7].reshape(1, 4))[0, i]) for i in range(3)) +
                     ''.join([',' + qpos_str[i] for i in self.motor_idx]) + '\n')

        self.fref.write(str(time) +
                     ''.join([',' + ref_joints_str[i] for i in range(10)]) + '\n')


def main():
    """
    restore latest model from ckpt
    """
    model_dir = "../../tf_models/" + args.model
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    model_path = latest_checkpoint
    print(model_dir, model_path)
    if args.data:
        writer = Writecsv(args.data)

    max_timestep = 900
    switch_interval = 100
    period = 13
    x = np.linspace(0, 2*np.pi, num=period)
    # bounds = np.array([[-1, 1],
    #                    [-0.25, 0.25],
    #                    [0.9, 0.65]])
    # vx_max = 0.6
    # vy_max = 0.25
    # cosx = np.cos(x)
    # sinx = np.sin(x)


    pi = train(max_iters=1, ref_name=args.ref)
    U.load_state(model_path)
    env = DigitEnv(max_timesteps=max_timestep,
                    ref_file="../motions/" + args.ref,
                    is_visual=True)
    draw_state = env.render()
    curr_ref_gait_idx = 0

    while draw_state:

        ob_vf, ob_pol = env.reset(if_roll=False)
        draw_state = env.render()
        time.sleep(1)

        switch_cnt = 0
        while draw_state:
            if not env.vis.ispaused():
                ac = pi.act(stochastic=False, ob_vf=ob_vf, ob_pol=ob_pol)[0]
                ob_vf, ob_pol, reward, done, info = env.step(ac, restore=True)  # need modification
                draw_state = env.render()
                time.sleep(0.02)
                # print(env.height)
                if args.data:
                    writer.write(env.time_in_sec, env.qpos, env.qvel,
                                 curr_para)
                    if env.timestep == (switch_interval*period+1):
                        draw_state = False
                        break

                # from scipy.spatial.transform import Rotation as R
                # r = R.from_quat(np.concatenate([env.qvel[[3, 4, 5]], [0]]))
                # curr_rxyzdot = r.as_euler("xyz", degrees=True)
                # xyzdot = Quat2Rxyz(np.concatenate([[0], env.qvel[[3, 4, 5]]]).reshape(1,4))
                # print(np.rad2deg(env.qvel[[3, 4, 5]]), curr_rxyzdot, np.rad2deg(xyzdot))
                # print(np.sum(np.square(np.rad2deg(env.qvel[[3, 4, 5]]))),
                #       np.sum(np.square(curr_rxyzdot)), np.sum(np.square(np.rad2deg(xyzdot))), '\n')



                switch_cnt += 1
                if switch_cnt > switch_interval:
                    if curr_ref_gait_idx == ref_gait_paras.shape[0] - 1:
                        curr_ref_gait_idx = 0
                    else:
                        curr_ref_gait_idx += 1
                    print(np.around(env.gait_library.get_ref_gaitparams(env.time_in_sec), decimals=2))
                    switch_cnt = 0
                if done:
                    break

            else:
                while env.vis.ispaused() and draw_state:
                    draw_state = env.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, help="name of data csv file")
    parser.add_argument("-r", "--ref", type=str, default='GaitLibrary/GaitLibrary.gaitlib', help="reference motion name")
    # gaitlib-xyz-trial1 gaitlib-forward-trial7-cont6
    parser.add_argument("-m", "--model", type=str, default='dynarand-trial1', help="model folder name")
    # parser.add_argument("-r", "--ref", type=str, default='opt_gait_ref_100310_neg.csv', help="reference motion name")
    # parser.add_argument("-m", "--model", type=str, default='opt_gait_ref_100310_neg_trial1', help="model folder name")

    args = parser.parse_args()
    main()
