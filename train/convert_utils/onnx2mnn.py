import os
import sys
import onnx
import numpy as np

# Get model ID from command line or use default
ID = "rotor_thrusts2" if len(sys.argv) < 2 else sys.argv[1]

# Setup paths
path = f"../saved_models/saved_model_{ID}/best_model"
onnx_path = path + ".onnx"
mnn_path = path + ".mnn"

# Verify ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model verified successfully")

# Convert to MNN using MNNConvert
cmd = f"MNNConvert -f ONNX --modelFile {onnx_path} --MNNModel {mnn_path} --bizCode biz"
result = os.system(cmd)
print("MNN conversion results:", result)


