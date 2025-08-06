import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "yolov9-pytorch"))

from models.yolov9 import YOLOv9
from inference import load_image, run_inference


def test_inference_pipeline(tmp_path):
    img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
    img_path = tmp_path / "img.jpg"
    img.save(img_path)
    tensor = load_image(img_path, 256)
    assert tensor.shape == (1, 3, 256, 256)
    model = YOLOv9("n", num_classes=80)
    preds = run_inference(model, tensor)
    assert len(preds) == 3
    assert preds[0].shape == (1, 84, 32, 32)
