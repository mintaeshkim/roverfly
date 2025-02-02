import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


# Load s_record_setpoint and a_record_setpoint
s_record_files = [f'/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/s_record_setpoint_{val}.csv' for val in [0.0, 0.1, 0.2, 0.3, 0.41, 1.01, 1.21, 1.42]]
a_record_files = [f'/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/a_record_setpoint_{val}.csv' for val in [0.0, 0.1, 0.2, 0.3, 0.41, 1.01, 1.21, 1.42]]

# Load and process all s_record_setpoint and a_record_setpoint files
s_record_setpoint = [np.loadtxt(file, delimiter=',')[1:] for file in s_record_files]
a_record_setpoint = [np.loadtxt(file, delimiter=',')[1:] for file in a_record_files]

# Calculate δ_record_setpoint (δ_{t+1} = s_{t+1} - s_t)
δ_record_setpoint = [s_record[1:] - s_record[:-1] for s_record in s_record_setpoint]

# Merge all s_record_setpoint and a_record_setpoint into single arrays, excluding the last timestep for consistency
s_record_setpoint = np.concatenate([s_record[:-1] for s_record in s_record_setpoint])
a_record_setpoint = np.concatenate([a_record for a_record in a_record_setpoint])
δ_record_setpoint = np.concatenate([δ_record for δ_record in δ_record_setpoint])

# mean, std = np.mean(δ_record_setpoint), np.std(δ_record_setpoint)
# δ_record_setpoint = (δ_record_setpoint - std) / mean

class QuadrotorDataset(Dataset):
    def __init__(self, s_record, a_record, δ_record):
        super(QuadrotorDataset, self).__init__()
        self.s_record = th.tensor(s_record, dtype=th.float32)  # (2000, 18)
        self.a_record = th.tensor(a_record, dtype=th.float32)  # (2000, 18)
        self.δ_record = th.tensor(δ_record, dtype=th.float32)  # (2000, 18)
    
    def __len__(self):
        return len(self.a_record)
    
    def __getitem__(self, idx):
        s_t = self.s_record[idx]
        a_t = self.a_record[idx]
        δ_t_plus_1 = self.δ_record[idx]
        return s_t, a_t, δ_t_plus_1

# Define the model
class SetpointDeltaPredictor(nn.Module):
    def __init__(self, input_dim=22, output_dim=18):
        super(SetpointDeltaPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)  # Output \hat{\delta}_{t+1}
        )
    
    def forward(self, s_t, a_t):
        inputs = th.cat([s_t, a_t], dim=1)
        return self.net(inputs)

# Loss function
def loss_fn(δ, δ_hat):
    # loss = ||delta_{t+1} - \hat{delta}_{t+1}||
    return th.norm(δ - δ_hat, dim=-1).mean()

# Instantiate model and optimizer
model = SetpointDeltaPredictor()
optimizer = Adam(model.parameters(), lr=3e-4)

# Create dataset and dataloader for mini-batch training
dataset = QuadrotorDataset(s_record_setpoint, a_record_setpoint, δ_record_setpoint)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    
    epoch_loss = 0
    for s_batch, a_batch, δ_batch in dataloader:
        δ_hat = model(s_batch, a_batch)
        loss = loss_fn(δ_batch, δ_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")

# Save the model's state_dict (weights)
th.save(model.state_dict(), '/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/model_weights.pth')

# Test a datapoint

# 1. Set the model to evaluation mode
model.eval()

# 2. Prepare a single test sample (s_t and a_t)
rand_idx = [500 * i for i in range(30)]

for idx in rand_idx:
    s_t_test = s_record_setpoint[idx]
    s_t_plus_1_test = s_record_setpoint[idx+1]
    a_t_test = a_record_setpoint[idx]
    δ_t_plus_1_test = δ_record_setpoint[idx]

    # Convert to Tensor (the model expects torch.Tensor as input)
    s_t_tensor = th.tensor(s_t_test, dtype=th.float32).unsqueeze(0)  # Shape: (1, 18)
    a_t_tensor = th.tensor(a_t_test, dtype=th.float32).unsqueeze(0)  # Shape: (1, 18)

    # 3. Pass the data through the model (using the trained model)
    with th.no_grad():  # No gradient calculation during inference
        δ_hat = model(s_t_tensor, a_t_tensor).detach().numpy()

    # 4. Display the predicted result
    # Normalization
    # print(f"Normalized predicted δ_t+1: {δ_hat}")
    # print(f"Normalized ground truth δ_t+1: {δ_t_plus_1_test}")
    # print(f"Normalized error: {δ_t_plus_1_test - δ_hat}")
    # print()
    # print(f"Predicted δ_t+1: {(δ_hat + std) * mean}")
    # print(f"Ground truth δ_t+1: {(δ_t_plus_1_test + std) * mean}")
    # print(f"Error: {(δ_t_plus_1_test - δ_hat) * mean}")
    # print()
    # print(f"Predicted s_t+1: {s_t_test + (δ_hat + std) * mean}")
    # print(f"Ground truth s_t+1: {s_t_plus_1_test}")
    # print(f"Error: {s_t_plus_1_test - (s_t_test + (δ_hat + std) * mean)}")
    # print()

    # No normalization
    print(f"Normalized predicted δ_t+1: {δ_hat}")
    print(f"Normalized ground truth δ_t+1: {δ_t_plus_1_test}")
    print(f"Normalized error: {δ_t_plus_1_test - δ_hat}")
    print()
    print(f"Predicted δ_t+1: {δ_hat}")
    print(f"Ground truth δ_t+1: {δ_t_plus_1_test}")
    print(f"Error: {δ_t_plus_1_test - δ_hat}")
    print()
    print(f"Predicted s_t+1: {s_t_test + δ_hat}")
    print(f"Ground truth s_t+1: {s_t_plus_1_test}")
    print(f"Error: {s_t_plus_1_test - (s_t_test + δ_hat)}")
    print()


