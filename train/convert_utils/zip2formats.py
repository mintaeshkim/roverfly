import os, sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(parent_dir))

import torch as th
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=True)


ID = "test2"
# Example: model = PPO("MlpPolicy", "Pendulum-v1")
# PPO("MlpPolicy", "Pendulum-v1").save("PathToTrainedModel")
# model = PPO.load("saved_models/saved_model_"+ID+"/best_model.zip", device="cpu")
path = "../saved_models/saved_model_" + ID + "/best_model"
model_path = path + ".zip"
onnx_path = path + ".onnx"
mnn_path = path + ".mnn"

model = PPO.load(model_path, device="cpu")

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)
th.onnx.export(
    onnx_policy,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
)

##### Load and test with onnx

import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size)).astype(np.float32)
ort_sess = ort.InferenceSession(onnx_path)
actions, values, log_prob = ort_sess.run(None, {"input": observation})

print("ONNX test results:")
print("Actions:", actions)
print("Values:", values)
print("Log prob:", log_prob)

# Check that the predictions are the same with original model
with th.no_grad():
    th_actions, th_values, th_log_prob = model.policy(th.as_tensor(observation), deterministic=True)
    print("\nPyTorch model results:")
    print("Actions:", th_actions.numpy())
    print("Values:", th_values.numpy())
    print("Log prob:", th_log_prob.numpy())

##### Convert to MNN and test

# Convert ONNX to MNN
cmd = f"MNNConvert -f ONNX --modelFile {onnx_path} --MNNModel {mnn_path} --bizCode biz"
result = os.system(cmd)

print("\nMNN conversion results:", result)

# Test MNN model
import MNN
import MNN.expr as F

# Create interpreter
interpreter = MNN.Interpreter(mnn_path)
session = interpreter.createSession()

# Get input tensor
input_tensor = interpreter.getSessionInput(session)

# Create test input
test_input = observation  # Use same test input as ONNX
tmp_input = MNN.Tensor(test_input.shape, MNN.Halide_Type_Float,
                       test_input, MNN.Tensor_DimensionType_Caffe)
input_tensor.copyFrom(tmp_input)

# Run inference
interpreter.runSession(session)

# Get output tensors
outputs = []
output_names = ['38', '65', '67']  # Output tensor names

for name in output_names:
    tensor = interpreter.getSessionOutput(session, name)
    shape = tensor.getShape()
    print(f"Output tensor {name} shape:", shape)

    # Create output tensor with correct shape and type
    tmp_output = MNN.Tensor(
        shape,
        MNN.Halide_Type_Float,
        np.zeros(shape, dtype=np.float32),
        MNN.Tensor_DimensionType_Tensorflow
    )

    # Copy data from MNN tensor to numpy array
    tensor.copyToHostTensor(tmp_output)
    output_data = np.array(tmp_output.getData(), dtype=np.float32).reshape(shape)
    outputs.append(output_data)

print("\nMNN test results:")
print("Actions:", outputs[2])
print("Values:", outputs[0])
print("Log prob:", outputs[1])

print("MNN model loaded and tested successfully!")


print(f"\nConverted models saved at:")
print(f"ONNX: {onnx_path}")
print(f"MNN: {mnn_path}")