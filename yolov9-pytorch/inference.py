"""Run inference with a YOLOv9 model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from models.yolov9 import YOLOv9


def load_image(path: Path, img_size: int) -> torch.Tensor:
    """Load image file and convert to tensor.

    The image is resized to ``img_size`` and normalised to ``[0, 1]``.
    """

    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


@torch.no_grad()
def run_inference(model: YOLOv9, img: torch.Tensor) -> List[torch.Tensor]:
    """Forward ``img`` through ``model`` in evaluation mode."""

    model.eval()
    return model(img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv9 inference on an image")
    parser.add_argument("image", type=Path, help="path to the image file")
    parser.add_argument("--weights", type=Path, help="optional path to model weights")
    parser.add_argument("--variant", type=str, default="n", help="model variant (n/s/m/l)")
    parser.add_argument("--num-classes", type=int, default=80, help="number of model classes")
    parser.add_argument("--img-size", type=int, default=256, help="inference image size")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args
    if args.weights:
        model = YOLOv9.from_pretrained(str(args.weights), num_classes=args.num_classes, variant=args.variant)
    else:
        model = YOLOv9(variant=args.variant, num_classes=args.num_classes)
    img = load_image(args.image, args.img_size)
    preds = run_inference(model, img)
    for i, p in enumerate(preds):
        print(f"output[{i}] shape: {tuple(p.shape)}")


if __name__ == "__main__":
    main()

