import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        self.obs_curr_dim = 22
        self.history_len = 5
        self.future_len = 3
        self.e_dim = 6
        self.a_dim = 4
        self.ff_dim = 6
        self.e_history_compressed_dim = 12
        self.a_history_compressed_dim = 8
        self.ff_compressed_dim = 12

        features_dim = self.obs_curr_dim + self.e_history_compressed_dim + self.a_history_compressed_dim + self.ff_compressed_dim
        # features_dim = 54

        super().__init__(observation_space, features_dim)

        # 1D CNN for e history
        self.e_history_layer1 = torch.nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, stride=2)
        self.e_history_act1 = torch.nn.SiLU()
        self.e_history_layer2 = torch.nn.Conv1d(in_channels=12, out_channels=6, kernel_size=2, stride=1)

        # 1D CNN for a history
        self.a_history_layer1 = torch.nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2)
        self.a_history_act1 = torch.nn.SiLU()
        self.a_history_layer2 = torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=2, stride=1)

        # 1D CNN for feedforward
        self.ff_layer1 = torch.nn.Conv1d(in_channels=6, out_channels=12, kernel_size=1, stride=2)
        self.ff_act1 = torch.nn.SiLU()
        self.ff_layer2 = torch.nn.Conv1d(in_channels=12, out_channels=6, kernel_size=1, stride=1)

    def forward(self, obs_full):
        obs_curr = obs_full[:, :self.obs_curr_dim]
        e_history = obs_full[:, self.obs_curr_dim : self.obs_curr_dim + self.e_dim * self.history_len].reshape(-1, self.e_dim, self.history_len)
        a_history = obs_full[:, self.obs_curr_dim + self.e_dim * self.history_len : self.obs_curr_dim + self.e_dim * self.history_len + self.a_dim * self.history_len].reshape(-1, self.a_dim, self.history_len)
        ff = obs_full[:, self.obs_curr_dim + self.e_dim * self.history_len + self.a_dim * self.history_len :].reshape(-1, self.ff_dim, self.future_len)

        # Process e history
        e_history = self.e_history_act1(self.e_history_layer1(e_history))
        e_history = self.e_history_layer2(e_history)
        e_history = torch.flatten(e_history, start_dim=1)

        # Process a history
        a_history = self.a_history_act1(self.a_history_layer1(a_history))
        a_history = self.a_history_layer2(a_history)
        a_history = torch.flatten(a_history, start_dim=1)

        # Process ff
        ff = self.ff_act1(self.ff_layer1(ff))
        ff = self.ff_layer2(ff)
        ff = torch.flatten(ff, start_dim=1)

        obs = torch.cat([obs_curr, e_history, a_history, ff], axis=1)

        return obs