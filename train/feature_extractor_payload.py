import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-th.log(th.tensor(10000.0)) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(self, x):
        # x shape: (B, T, D)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        self.history_len = 5
        self.future_len = 5
        self.s_dim = 24
        self.d_dim = 12
        self.a_dim = 3
        self.ff_dim = 12
        self.obs_curr_dim = 41
        self.e_dim = self.s_dim + self.d_dim  # == 36

        self.embed_dim = 32
        self.num_heads = 4
        self.num_layers = 2

        self.s_history_compressed_dim = self.embed_dim
        self.a_history_compressed_dim = self.embed_dim
        self.ff_compressed_dim = self.embed_dim

        features_dim = self.obs_curr_dim + self.s_history_compressed_dim + self.a_history_compressed_dim + self.ff_compressed_dim

        super().__init__(observation_space, features_dim)

        self.e_linear = nn.Linear(self.e_dim, self.embed_dim)
        self.a_linear = nn.Linear(self.a_dim, self.embed_dim)
        self.ff_linear = nn.Linear(self.ff_dim, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, batch_first=True)
        self.e_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.a_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.ff_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.e_pos_encoder = PositionalEncoding(self.embed_dim, max_len=self.history_len)
        self.a_pos_encoder = PositionalEncoding(self.embed_dim, max_len=self.history_len)
        self.ff_pos_encoder = PositionalEncoding(self.embed_dim, max_len=self.history_len)

    def forward(self, obs_full):
        B = obs_full.shape[0]
        obs_curr = obs_full[:, :self.obs_curr_dim]

        # === Unpack and reshape sequences ===
        idx = self.obs_curr_dim
        e_history = obs_full[:, idx:idx + self.e_dim * self.history_len].reshape(B, self.history_len, self.e_dim)
        idx += self.e_dim * self.history_len
        a_history = obs_full[:, idx:idx + self.a_dim * self.history_len].reshape(B, self.history_len, self.a_dim)
        idx += self.a_dim * self.history_len
        ff = obs_full[:, idx:idx + self.ff_dim * self.history_len].reshape(B, self.history_len, self.ff_dim)

        # === Linear projection to embedding space ===
        e_emb = self.e_linear(e_history)
        a_emb = self.a_linear(a_history)
        ff_emb = self.ff_linear(ff)

        # === Positional Encoding ===
        e_emb = self.e_pos_encoder(e_emb)
        a_emb = self.a_pos_encoder(a_emb)
        ff_emb = self.ff_pos_encoder(ff_emb)

        # === Transformer Encoder ===
        e_out = self.e_transformer(e_emb)[:, -1, :]  # Use last time step
        a_out = self.a_transformer(a_emb)[:, -1, :]
        ff_out = self.ff_transformer(ff_emb)[:, -1, :]

        # === Concatenate everything ===
        features = th.cat([obs_curr, e_out, a_out, ff_out], dim=1)

        return features
