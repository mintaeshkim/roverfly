"""Export Stable-Baselines3 policies for deployment."""

import shutil
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

ExportFormat = Literal["onnx", "pytorch", "mnn"]


class OnnxPolicy(torch.nn.Module):
    """Thin deterministic wrapper around an SB3 actor-critic policy."""

    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor):
        return self.policy(observation, deterministic=True)


def export_policy(
    model_path: Path,
    *,
    output_path: Path | None = None,
    format: ExportFormat = "onnx",
    opset: int = 17,
) -> Path:
    """Export a saved PPO policy to ONNX, a state dict, or MNN."""
    model_path = model_path.expanduser().resolve()
    model = PPO.load(str(model_path), device="cpu")
    suffix = {"onnx": ".onnx", "pytorch": ".pth", "mnn": ".mnn"}[format]
    output_path = (output_path or model_path.with_suffix(suffix)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pytorch":
        torch.save(model.policy.state_dict(), output_path)
        return output_path

    onnx_path = output_path if format == "onnx" else output_path.with_suffix(".onnx")
    observation_shape = model.observation_space.shape
    if observation_shape is None:
        raise ValueError("Only fixed-shape observation spaces can be exported")
    dummy_input = torch.zeros((1, *observation_shape), dtype=torch.float32)
    torch.onnx.export(
        OnnxPolicy(model.policy),
        dummy_input,
        str(onnx_path),
        opset_version=opset,
        input_names=["observation"],
        output_names=["action", "value", "log_probability"],
        dynamo=False,
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
            "value": {0: "batch"},
            "log_probability": {0: "batch"},
        },
    )
    verify_onnx(onnx_path, model=model)

    if format == "mnn":
        converter = shutil.which("MNNConvert")
        if converter is None:
            raise RuntimeError("MNNConvert is required to export an MNN model")
        subprocess.run(
            [
                converter,
                "-f",
                "ONNX",
                "--modelFile",
                str(onnx_path),
                "--MNNModel",
                str(output_path),
                "--bizCode",
                "roverfly",
            ],
            check=True,
        )
    return output_path


def verify_onnx(onnx_path: Path, *, model: PPO | None = None) -> None:
    """Validate an ONNX graph and optionally compare it with its SB3 policy."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install RoVerFly with the 'export' extra") from exc

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    if model is None:
        return

    observation_shape = model.observation_space.shape
    observation = np.zeros((1, *observation_shape), dtype=np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"observation": observation})
    with torch.no_grad():
        expected = model.policy(torch.as_tensor(observation), deterministic=True)
    for onnx_value, torch_value in zip(actual, expected):
        np.testing.assert_allclose(onnx_value, torch_value.cpu().numpy(), rtol=1e-4, atol=1e-5)
