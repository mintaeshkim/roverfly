import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
from envs.quadrotor_env import QuadrotorEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.ppo.ppo import PPO # Customized
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments')
    
    # Execution parameters
    parser.add_argument('--id', type=str, default='untitled', help='Provide experiment name and ID.')
    parser.add_argument('--visualize', type=bool, default=False, help='Choose visualization option.')
    
    args = parser.parse_args()
    args_dict = vars(args)

    return args_dict

def main():
    args_dict = parse_arguments()
    print(args_dict)
    
    # Path
    experiment_id = args_dict['id']
    save_path = os.path.join('saved_models/saved_model_'+experiment_id)
    loaded_model = PPO.load(save_path+"/best_model")
    
    # Environment parameters
    render_mode = 'human' if args_dict['visualize'] else None

    env = QuadrotorEnv(render_mode=render_mode)
    env = VecMonitor(DummyVecEnv([lambda: env]))
    
    print("Evaluation start!")
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=50, render=render_mode)
    env.close()
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

if __name__ == "__main__":
    main()