import sys
import os
sys.path.append("..")
import envs
from mpi4py import MPI
import numpy as np
# from stable_baselines3 import logger
# from stable_baselines3.common import tf_util as U
import tensorflow as tf
import ppo_sgd_simple_mlp


# ------------------------------------
render = False
saved_model = 'baseline_hover_test'
# ------------------------------------

os.environ["OPENAI_LOGDIR"] = "../logs/" + saved_model
os.environ["OPENAI_LOG_FORMAT"] = "stdout,log,tensorboard"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_folder = "../tf_model/"
restore_model_from_file = None

def train(max_iters, with_gpu=False, callback=None):
    # training define
    from Flappy_Integrated.flappy.envs.train.run import mlp_policy

    if not with_gpu:
        config = tf.ConfigProto(device_count={"GPU": 0})
        # U.make_session(config=config).__enter__()
        print("**************Using CPU**************")
    else:
        # U.make_session().__enter__()
        print("**************Using GPU**************")

    def policy_fn(name, ob_space_vf, ob_space_pol, ac_space):
        return mlp_policy.MlpPolicy(
            name=name,
            ob_space_vf=ob_space_vf,
            ob_space_pol=ob_space_pol,
            ac_space=ac_space,
            hid_size=512,
            num_hid_layers=2,
        )

    env = envs.FlappingWingEnv(max_timesteps=750,
                                is_visual=render,
                                randomize=False,
                                debug=True,lpf_action=False,
                                traj_type=False)

    pi = ppo_sgd_simple_mlp.learn(
        env,
        policy_fn,
        max_iters=max_iters,
        timesteps_per_actorbatch=2048,  # 4096 512
        clip_param=0.2,
        entcoeff=0,  # 0.2 0.0
        optim_epochs=2,  # 4
        optim_stepsize=1e-4,  # 3e-5,
        optim_batchsize=256,  # 512 256
        gamma=0.98,  # 0.99 # look forward 1.65s
        lam=0.95,
        callback=callback,
        schedule="constant",
        continue_from=restore_model_from_file
    )
    return pi


def callback(locals_, globals_):
    saver_ = locals_["saver"]
    # sess_ = U.get_session()
    timesteps_so_far_ = locals_["timesteps_so_far"]
    iters_so_far_ = locals_["iters_so_far"]
    model_dir = model_folder + saved_model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if MPI.COMM_WORLD.Get_rank() == 0 and iters_so_far_ % 30 == 0:
        saver_.save(sess_, model_dir + "/model", global_step=timesteps_so_far_)
    return True


if __name__ == "__main__":
    # logger.configure()
    train(max_iters=8000, callback=callback)
