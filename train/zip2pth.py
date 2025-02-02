import os, sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))
sys.path.append(os.path.join(parent_dir, 'envs'))
from envs.quadrotor_payload_env import QuadrotorPayloadEnv

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from train.feature_extractor_payload import CustomFeaturesExtractor


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs,
                                                      features_extractor_class=CustomFeaturesExtractor)


# model = PPO.load("saved_models/saved_model_test_payload_1/best_model.zip")
# policy = model.policy

# torch.save(policy.state_dict(), "saved_models/saved_model_test_payload_1/payload_policy_state_dict.pth")

env = QuadrotorPayloadEnv()

model = PPO(CustomActorCriticPolicy, env=env)
state_dict = torch.load("saved_models/saved_model_test_payload_1/payload_policy_state_dict.pth")
model.policy.load_state_dict(state_dict)

obs = np.zeros((1, 320))
action, _states = model.predict(obs)