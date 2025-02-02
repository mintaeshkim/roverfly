import numpy as np
import torch as th
import torch.nn as nn

# Load s_record_curve and a_record_curve
s_record_files = [f'/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/record_curve/s_record_curve_{val}.csv' for val in [1.34, 1.49, 1.57, 1.67]]
a_record_files = [f'/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/record_curve/a_record_curve_{val}.csv' for val in [1.34, 1.49, 1.57, 1.67]]

# Load and process all s_record_curve and a_record_curve files
s_record_curve_list = [np.loadtxt(file, delimiter=',')[1:] for file in s_record_files]
a_record_curve_list = [np.loadtxt(file, delimiter=',')[1:] for file in a_record_files]

# Prepare inputs and outputs for testing
inputs_list = []
outputs_list = []

for s_record, a_record in zip(s_record_curve_list, a_record_curve_list):
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

# Test the model

# 1. Set the model to evaluation mode
# Define the model
class CurveDeltaPredictor(nn.Module):
    def __init__(self, input_dim=66, output_dim=18):
        super(CurveDeltaPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # Output \hat{\delta}_{t+1}
        )
        
    def forward(self, x):
        return self.net(x)

model = CurveDeltaPredictor()
model.load_state_dict(th.load('/Users/mintaekim/Desktop/HRL/Quadrotor/quadrotor_v1/train/data/record_curve/model_weights_curve_multiple.pth'))
print("Model loaded!")
model.eval()

# 2. Test on random samples
# You can test on 16 random samples
random_indices = th.randint(0, len(inputs_tensor), (16,))  # Generate 16 random indices
for idx in random_indices:
    inputs_test = inputs_tensor[idx]
    delta_t_plus_1_test = outputs_tensor[idx]
    
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
    print(f"Predicted δ_{idx%2000+1}: {delta_hat}")
    print(f"Ground truth δ_{idx%2000+1}: {delta_t_plus_1_test}")
    print(f"Error in δ_{idx%2000+1}: {delta_t_plus_1_test - delta_hat}\n")

    print(f"Predicted s_{idx%2000+1}: {s_t_plus_1_pred}")
    print(f"Ground truth s_{idx%2000+1}: {s_t_plus_1_test}")
    print(f"Error in s_{idx%2000+1}: {s_t_plus_1_test - s_t_plus_1_pred}\n")
