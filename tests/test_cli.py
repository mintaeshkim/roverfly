from roverfly.cli import build_parser


def test_train_arguments_use_standard_boolean_flag() -> None:
    args = build_parser().parse_args(["train", "--env", "payload", "--visualize"])
    assert args.env == "payload"
    assert args.visualize is True


def test_export_arguments() -> None:
    args = build_parser().parse_args(["export", "model.zip", "--format", "onnx"])
    assert args.format == "onnx"
