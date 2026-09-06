"""Command-line interface for training and evaluating RoVerFly policies."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from roverfly.envs import ENVIRONMENTS
from roverfly.training.config import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roverfly", description="Train and evaluate RoVerFly policies."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a PPO policy")
    train_parser.add_argument("--env", choices=ENVIRONMENTS, default="mini")
    train_parser.add_argument("--id", dest="experiment", default="untitled")
    train_parser.add_argument("--num-envs", type=int, default=8)
    train_parser.add_argument("--num-steps", dest="total_timesteps", type=int, default=100_000_000)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument("--visualize", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="evaluate a saved PPO policy")
    eval_parser.add_argument("model", type=Path)
    eval_parser.add_argument("--env", choices=ENVIRONMENTS, default="mini")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--visualize", action="store_true")

    export_parser = subparsers.add_parser("export", help="export a PPO policy for deployment")
    export_parser.add_argument("model", type=Path)
    export_parser.add_argument("--format", choices=("onnx", "pytorch", "mnn"), default="onnx")
    export_parser.add_argument("--output", type=Path)
    export_parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from roverfly.training.runner import train

        config = TrainingConfig(
            environment=args.env,
            experiment=args.experiment,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            visualize=args.visualize,
        )
        model_path = train(config)
        print(f"Saved final model to {model_path}")
        return 0

    if args.command == "export":
        from roverfly.export import export_policy

        output_path = export_policy(
            args.model,
            output_path=args.output,
            format=args.format,
            opset=args.opset,
        )
        print(f"Exported policy to {output_path}")
        return 0

    from roverfly.evaluation import evaluate

    mean, std = evaluate(
        args.model,
        environment=args.env,
        episodes=args.episodes,
        device=args.device,
        visualize=args.visualize,
    )
    print(f"Mean reward: {mean:.3f} +/- {std:.3f}")
    return 0
