"""Export a YOLOv9 model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.yolov9 import YOLOv9


def export_onnx(
    out_path: Path,
    weights: Path | None = None,
    variant: str = "n",
    num_classes: int = 80,
    img_size: int = 256,
) -> None:
    """Export the model as an ONNX file."""

    if weights:
        model = YOLOv9.from_pretrained(str(weights), num_classes=num_classes, variant=variant)
    else:
        model = YOLOv9(variant=variant, num_classes=num_classes)
    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=12,
        input_names=["images"],
        output_names=["preds"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv9 model to ONNX")
    parser.add_argument("out", type=Path, help="output ONNX file path")
    parser.add_argument("--weights", type=Path, help="optional path to model weights")
    parser.add_argument("--variant", type=str, default="n")
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--img-size", type=int, default=256)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    export_onnx(args.out, args.weights, args.variant, args.num_classes, args.img_size)
    print(f"ONNX model saved to {args.out}")


if __name__ == "__main__":
    main()

