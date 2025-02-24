import os
import numpy as np
import torch as th
import onnxruntime as ort
import MNN
from stable_baselines3 import PPO


def load_and_infer_zip(model_path, observation):
    """Load ZIP model and run inference"""
    print("\n=== ZIP Model Inference ===")
    model = PPO.load(model_path, device="cpu")

    # Convert numpy observation to torch tensor
    obs_tensor = th.as_tensor(observation)

    with th.no_grad():
        actions, values, log_prob = model.policy(obs_tensor, deterministic=True)

    # Convert outputs to numpy for comparison
    return {
        'actions': actions.numpy(),
        'values': values.numpy(),
        'log_prob': log_prob.numpy()
    }


def load_and_infer_onnx(onnx_path, observation):
    """Load ONNX model and run inference"""
    print("\n=== ONNX Model Inference ===")
    ort_sess = ort.InferenceSession(onnx_path)

    # Run inference
    outputs = ort_sess.run(None, {"input": observation})

    return {
        'actions': outputs[0],
        'values': outputs[1],
        'log_prob': outputs[2]
    }


def load_and_infer_mnn(mnn_path, observation):
    """Load MNN model and run inference"""
    print("\n=== MNN Model Inference ===")

    try:
        # Create interpreter and session
        interpreter = MNN.Interpreter(mnn_path)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)

        # Print shapes for debugging
        print("Input observation shape:", observation.shape)
        print("MNN input tensor shape:", input_tensor.getShape())

        # Ensure the input is contiguous and correct type
        observation = np.ascontiguousarray(observation, dtype=np.float32)

        # Create input tensor
        tmp_input = MNN.Tensor(
            observation.shape,
            MNN.Halide_Type_Float,
            observation,
            MNN.Tensor_DimensionType_Tensorflow
        )

        # Copy data to input tensor
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

        return {
            'actions': outputs[2],
            'values': outputs[0],
            'log_prob': outputs[1]
        }

    except Exception as e:
        print("Error during MNN inference:", str(e))
        return None


def compare_outputs(zip_out, onnx_out, mnn_out):
    """Compare outputs from all three models"""
    print("\n=== Output Comparison ===")

    if mnn_out is None:
        print("Cannot compare - MNN inference failed")
        return False

    # Compare actions
    print("\nActions comparison:")
    print("ZIP:", zip_out['actions'])
    print("ONNX:", onnx_out['actions'])
    print("MNN:", mnn_out['actions'])

    actions_match = (
            np.allclose(zip_out['actions'], onnx_out['actions'], rtol=1e-4) and
            np.allclose(zip_out['actions'], mnn_out['actions'], rtol=1e-4)
    )
    print("Actions match:", actions_match)

    # Compare values
    print("\nValues comparison:")
    print("ZIP:", zip_out['values'])
    print("ONNX:", onnx_out['values'])
    print("MNN:", mnn_out['values'])

    values_match = (
            np.allclose(zip_out['values'], onnx_out['values'], rtol=1e-4) and
            np.allclose(zip_out['values'], mnn_out['values'], rtol=1e-4)
    )
    print("Values match:", values_match)

    # Compare log probabilities
    print("\nLog probabilities comparison:")
    print("ZIP:", zip_out['log_prob'])
    print("ONNX:", onnx_out['log_prob'])
    print("MNN:", mnn_out['log_prob'])

    log_prob_match = (
            np.allclose(zip_out['log_prob'], onnx_out['log_prob'], rtol=1e-4) and
            np.allclose(zip_out['log_prob'], mnn_out['log_prob'], rtol=1e-4)
    )
    print("Log probabilities match:", log_prob_match)

    return all([actions_match, values_match, log_prob_match])


if __name__ == "__main__":
    # Model paths
    ID = "test"  # Change this to your model ID
    path = f"../saved_models/saved_model_{ID}/best_model"
    zip_path = path + ".zip"
    onnx_path = path + ".onnx"
    mnn_path = path + ".mnn"

    # Load the ZIP model first to get observation shape
    model = PPO.load(zip_path, device="cpu")
    observation_size = model.observation_space.shape
    del model  # Free memory

    # Create test input
    observation = np.zeros((1, *observation_size)).astype(np.float32)

    # Run inference with all three models
    zip_output = load_and_infer_zip(zip_path, observation)
    onnx_output = load_and_infer_onnx(onnx_path, observation)
    mnn_output = load_and_infer_mnn(mnn_path, observation)

    # Compare outputs
    all_match = compare_outputs(zip_output, onnx_output, mnn_output)

    print("\n=== Final Result ===")
    if all_match:
        print("Success! All model outputs match within tolerance.")
    else:
        print("Warning: Some model outputs don't match. Check the comparisons above.")