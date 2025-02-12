import os
import sys
import numpy as np
import torch as th
import onnx
import onnxruntime as ort
from stable_baselines3 import PPO

# 환경 import
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
from envs.quadrotor_fb_env import QuadrotorEnv

# PPO 모델 초기화
activation_fn = th.nn.Tanh
net_arch = {'pi': [128, 128], 'vf': [128, 128]}
env = QuadrotorEnv()

model = PPO('MlpPolicy',
            env=env,
            policy_kwargs={'activation_fn': activation_fn, 'net_arch': net_arch})

# PTH 파일 로드
pth_path = "saved_models/saved_model_quadrotor_hover_100hz_1/best_model/policy.pth"
state_dict = th.load(pth_path, map_location=th.device('cpu'))
model.policy.load_state_dict(state_dict)

# 입력 데이터 준비 (더미 관측값)
obs = np.array([[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]]).astype(np.float32)

# PyTorch 모델 예측
action_pth, _ = model.predict(obs)
print("PTH 모델 예측 결과:", action_pth)





import torch as th
import torch.nn as nn
import numpy as np

# Define the SB3 PPO Policy Network correctly
class SB3PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SB3PPOPolicy, self).__init__()

        # Feature extractor (shared by policy and value networks)
        self.mlp_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # Policy network (actor)
        self.action_net = nn.Linear(128, output_dim)

        # Log standard deviation (learnable)
        self.log_std = nn.Parameter(th.zeros(output_dim))

    def forward(self, x, deterministic=False):
        """
        Compute the action given the observation.

        :param x: Input observation tensor
        :param deterministic: If True, return mean action; otherwise, sample from the Gaussian distribution.
        :return: Action tensor
        """
        latent = self.mlp_extractor(x)  # Extract features
        mean_action = self.action_net(latent)  # Compute action mean
        std = th.exp(self.log_std)  # Convert log_std to standard deviation

        if deterministic:
            return mean_action  # Deterministic mode (mean action)
        else:
            return mean_action + std * th.randn_like(std)  # Sampled action

# Load the saved SB3 model state_dict
pth_path = "saved_models/saved_model_quadrotor_hover_100hz_1/best_model/policy.pth"
state_dict = th.load(pth_path, map_location=th.device('cpu'))

# Extract only the required keys from SB3 state_dict
policy_state_dict = {
    k.replace("mlp_extractor.policy_net.", ""): v
    for k, v in state_dict.items() if "mlp_extractor.policy_net" in k
}

# Extract action_net weights
policy_state_dict["action_net.weight"] = state_dict["action_net.weight"]
policy_state_dict["action_net.bias"] = state_dict["action_net.bias"]

# Extract log_std
if "log_std" in state_dict:
    log_std_value = state_dict["log_std"]
else:
    raise KeyError("log_std not found in state_dict!")

# Define input/output dimensions
input_dim = 90  # Adjust based on observation space
output_dim = 4  # Adjust based on action space

# Initialize policy network and load extracted weights
policy = SB3PPOPolicy(input_dim, output_dim)
policy.load_state_dict(policy_state_dict, strict=False)  # Allow partial loading
policy.log_std.data = log_std_value  # Load log_std
policy.eval()  # Set to evaluation mode

# Prepare input observation (dummy observation)
obs = np.array([[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]]).astype(np.float32)

# Convert observation to PyTorch tensor
obs_tensor = th.tensor(obs, dtype=th.float32)

# Get action from the policy network
with th.no_grad():
    action = policy(obs_tensor)

print("Predicted Action:", action.numpy())
