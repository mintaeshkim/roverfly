import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

# Load s_record_setpoint and a_record_setpoint
s_record_files = [f'/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/record_setpoint/s_record_setpoint_{val}.csv' for val in [0.0, 0.1, 0.2, 0.3, 0.41, 1.01, 1.21, 1.42]]
a_record_files = [f'/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/record_setpoint/a_record_setpoint_{val}.csv' for val in [0.0, 0.1, 0.2, 0.3, 0.41, 1.01, 1.21, 1.42]]

# Load and process all s_record_setpoint and a_record_setpoint files
s_record_setpoint_list = [np.loadtxt(file, delimiter=',')[1:] for file in s_record_files]
a_record_setpoint_list = [np.loadtxt(file, delimiter=',')[1:] for file in a_record_files]

# Prepare inputs and outputs
inputs_list = []
outputs_list = []

for s_record, a_record in zip(s_record_setpoint_list, a_record_setpoint_list):
    delta_record = s_record[1:] - s_record[:-1]  # Compute delta_{t+1}

    s_record_th = th.tensor(s_record, dtype=th.float32)
    a_record_th = th.tensor(a_record, dtype=th.float32)
    delta_record_th = th.tensor(delta_record, dtype=th.float32)

    num_samples = len(s_record[:-1])
    length = num_samples - 5  # Adjusted for indices up to t-5

    for idx in range(length):
        t = idx + 5

        # Inputs: s_{t-5}, s_{t-4}, s_{t-3}; a_{t-5}, a_{t-4}, a_{t-3}
        s_t_minus_5 = s_record_th[t - 5]
        s_t_minus_4 = s_record_th[t - 4]
        s_t_minus_3 = s_record_th[t - 3]

        a_t_minus_5 = a_record_th[t - 5]
        a_t_minus_4 = a_record_th[t - 4]
        a_t_minus_3 = a_record_th[t - 3]

        inputs = th.cat([
            s_t_minus_5, s_t_minus_4, s_t_minus_3,
            a_t_minus_5, a_t_minus_4, a_t_minus_3
        ])

        # Output: δ_{t+1} = s_{t+1} - s_t
        delta_t_plus_1 = delta_record_th[t]

        inputs_list.append(inputs)
        outputs_list.append(delta_t_plus_1)

# Stack inputs and outputs
inputs_tensor = th.stack(inputs_list)  # Shape: (total_samples, 66)
outputs_tensor = th.stack(outputs_list)  # Shape: (total_samples, 18)

# Define the dataset
class QuadrotorDataset(Dataset):
    def __init__(self, inputs_tensor, outputs_tensor):
        self.inputs = inputs_tensor
        self.outputs = outputs_tensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# Create dataset and dataloader
dataset = QuadrotorDataset(inputs_tensor, outputs_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the model
class SetpointDeltaPredictor(nn.Module):
    def __init__(self, input_dim=66, output_dim=18):
        super(SetpointDeltaPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # Output \hat{\delta}_{t+1}
        )
        
    def forward(self, x):
        return self.net(x)

# Instantiate model and optimizer
model = SetpointDeltaPredictor()
optimizer = Adam(model.parameters(), lr=3e-4)

# Loss function
def loss_fn(delta, delta_hat):
    # loss = ||delta_{t+1} - \hat{delta}_{t+1}||
    return th.norm(delta - delta_hat, dim=-1).mean()

# Training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for inputs_batch, outputs_batch in dataloader:
        delta_hat = model(inputs_batch)
        loss = loss_fn(outputs_batch, delta_hat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")

# Save the model's state_dict (weights)
th.save(model.state_dict(), '/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/record_setpoint/model_weights_multiple.pth')

# Test a datapoint

# 1. Set the model to evaluation mode
model.eval()

# 2. Prepare a single test sample
rand_indices = [500 * i for i in range(30)]
for idx in rand_indices:
    if idx >= len(dataset):
        continue  # Skip if index is out of bounds

    inputs_test, delta_t_plus_1_test = dataset[idx]
    # Extract s_{t-3} from inputs (positions 36 to 54)
    s_t_minus_3 = inputs_test[2*18:3*18]
    s_t = s_t_minus_3  # s_t corresponds to s_{t-3} in this setup

    # Ground truth s_{t+1}
    s_t_plus_1_test = s_t + delta_t_plus_1_test

    # 3. Pass the data through the model
    with th.no_grad():
        delta_hat = model(inputs_test.unsqueeze(0)).squeeze(0)

    # Predicted s_{t+1}
    s_t_plus_1_pred = s_t + delta_hat

    # 4. Display the predicted result
    print(f"Predicted δ_{t+1}: {delta_hat}")
    print(f"Ground truth δ_{t+1}: {delta_t_plus_1_test}")
    print(f"Error in δ_{t+1}: {delta_t_plus_1_test - delta_hat}\n")

    print(f"Predicted s_{t+1}: {s_t_plus_1_pred}")
    print(f"Ground truth s_{t+1}: {s_t_plus_1_test}")
    print(f"Error in s_{t+1}: {s_t_plus_1_test - s_t_plus_1_pred}\n")
