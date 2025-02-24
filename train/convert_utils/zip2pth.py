import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))

import torch as th
from typing import Tuple
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from train.feature_extractor_payload import CustomFeaturesExtractor


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs,
                                                     features_extractor_class=CustomFeaturesExtractor)

ID = "test"
path = "../saved_models/saved_model_"+ID+"/best_model"
model_path = path+".zip"
pth_path = path+".pth"

# Load the ZIP model
model = PPO.load(model_path, device="cpu")

# Save policy state dict
th.save(model.policy.state_dict(), pth_path)

##### Load and test with pth

# Create test environment and model
from envs.quadrotor_env import QuadrotorEnv

env = QuadrotorEnv()
test_model = PPO(CustomActorCriticPolicy, env=env)

# Load the saved state dict
state_dict = th.load(pth_path)
test_model.policy.load_state_dict(state_dict)

# Test with dummy input
observation = np.zeros((1, 320)).astype(np.float32)
test_actions, _states = test_model.predict(observation)

print("\nTest results:")
print("Observation shape:", observation.shape)
print("Action output:", test_actions)

# Compare with original model
original_actions, original_states = model.predict(observation)
print("\nOriginal model output:", original_actions)
print("Outputs match:", np.allclose(test_actions, original_actions, rtol=1e-5))