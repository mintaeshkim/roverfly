import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
from envs.ppo.ppo import PPO # Customized
# from stable_baselines3 import PPO # Naive
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import numpy as np
from envs.quadrotor_payload_env import QuadrotorPayloadEnv
from train.feature_extractor_payload import CustomFeaturesExtractor
import argparse
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    
    # Execution parameters
    parser.add_argument('--id', type=str, default='untitled', help='Provide experiment name and ID.')
    parser.add_argument('--visualize', type=bool, default=False, help='Choose visualization option.')
    parser.add_argument('--device', type=str, default='mps', help='Provide device info.')
    parser.add_argument('--num_envs', type=int, default=8, help='Provide number of parallel environments.')
    parser.add_argument('--num_steps', type=int, default=1e+8, help='Provide number of steps.')

    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict


class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            episode_rewards = self.locals['episode']['r']
            for idx, reward in enumerate(episode_rewards):
                self.logger.record('reward/reward_{}'.format(idx), reward)
        return True

class EvalCallbackWithTimestamp(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward_prev = -np.inf

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.best_model_save_path is not None and self.best_mean_reward > self.best_mean_reward_prev:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            new_save_path = os.path.join(self.best_model_save_path, f"best_model_{timestamp}")
            self.model.save(new_save_path)
            print(f"New best model saved to {new_save_path}")
            self.best_mean_reward_prev = self.best_mean_reward
        return result

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs,
                                                      features_extractor_class=CustomFeaturesExtractor)

def main():
    args_dict = parse_arguments()
    print(args_dict)
    
    # Path
    experiment_id = args_dict['id']
    log_path = os.path.join('logs')
    save_path = os.path.join('saved_models/saved_model_'+experiment_id)
    
    # Environment parameters
    render_mode = 'human' if args_dict['visualize'] else None
    
    # Parallel environment
    def create_env(seed=0):
        def _init():
            env = QuadrotorPayloadEnv(render_mode=render_mode,
                                      env_num=seed)
            return env
        set_random_seed(seed)
        return _init
    
    num_envs = args_dict['num_envs']
    env = VecMonitor(DummyVecEnv([create_env(seed=i) for i in range(num_envs)]))

    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=59500, verbose=1)
    eval_callback = EvalCallbackWithTimestamp(env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=2500,
                                 best_model_save_path=save_path,
                                 verbose=1)
    
    reward_logging_callback = RewardLoggingCallback()
    callback_list = CallbackList([reward_logging_callback, eval_callback])
    
    # Networks
    # NOTE: if is_history: use 128 or more
    activation_fn = th.nn.Tanh
    # net_arch = {'pi': [64,64,64],
    #             'vf': [64,64,64]}
    # net_arch = {'pi': [128,96,64],
    #             'vf': [128,96,64]}
    # net_arch = {'pi': [256,128,64],
    #             'vf': [256,128,64]}
    # net_arch = {'pi': [512,512],
    #             'vf': [512,512]}
    net_arch = {'pi': [256,256],
                'vf': [256,256]}
    # net_arch = {'pi': [64,64],
    #             'vf': [64,64]}
    # net_arch = {'pi': [512,256,128],
    #             'vf': [512,256,128]}

    # PPO Modeling
    def linear_schedule(initial_value):
        if isinstance(initial_value, str):
            initial_value = float(initial_value)
        def func(progress):
            return progress * initial_value
        return func

    num_steps = args_dict['num_steps']
    device = args_dict['device']

    horizon_length = 64 if num_envs >= 16 else 16384
    n_steps = horizon_length
    batch_size = 32 * num_envs if num_envs >= 16 else 8192

    model = PPO('MlpPolicy',  # CustomActorCriticPolicy,
                env=env,
                learning_rate=1e-4,
                n_steps=n_steps, # 2048  |  The number of steps to run for each environment per update / 2048 if dt=0.001 / 2048*16 if dt=0.01
                batch_size=batch_size, # 512*num_cpu  |  *16 if dt=0.01
                gamma=0.99,  # 0.99 # look forward 1.65s
                gae_lambda=0.95,  # 0.95
                clip_range=linear_schedule(0.2),
                ent_coef=0.0, # Makes PPO explore 0.02
                verbose=1,
                policy_kwargs={'activation_fn':activation_fn, 'net_arch':net_arch}, # policy_kwargs={'activation': 'dual', 'thrust': 2.55, 'thrust_max': 5.0},
                tensorboard_log=log_path,
                device=device)

    model.learn(total_timesteps=num_steps, # The total number of samples (env steps) to train on
                progress_bar=True,
                callback=callback_list)

    model.save(save_path)


    ####################################################
    #################### Evaluation ####################
    ####################################################

    obs_sample = model.env.observation_space.sample()

    print("Pre saved model prediction: ")
    print(model.predict(obs_sample, deterministic=True))
    del model # delete trained model to demonstrate loading

    loaded_model = PPO.load(save_path+"/best_model")
    print("Loaded model prediction: ")
    print(loaded_model.predict(obs_sample, deterministic=True))

    print("Evaluation start")
    evaluate_policy(loaded_model, env, n_eval_episodes=5, render=False)
    env.close()

main()