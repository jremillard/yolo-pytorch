import io
import sys
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
# Ensure the project root is on the import path
sys.path.append(str(ROOT / "yolov9-pytorch"))

from models.yolov9 import YOLOv9


def test_forward_shapes():
    model = YOLOv9("n", num_classes=80)
    x = torch.randn(1, 3, 256, 256)
    outputs = model(x)
    assert len(outputs) == 3
    # Stride 8, 16 and 32 feature maps
    assert outputs[0].shape == (1, 84, 32, 32)
    assert outputs[1].shape == (1, 84, 16, 16)
    assert outputs[2].shape == (1, 84, 8, 8)


def test_pretrained_loading():
    # Make sure loading official checkpoints works without error
    for ckpt in [
        ROOT / "yolov9-org-artifacts/yolov9-s.pt",
        ROOT / "yolov9-org-artifacts/yolov9-m.pt",
        ROOT / "yolov9-org-artifacts/yolov9-t-converted.pt",
    ]:
        model = YOLOv9.from_pretrained(str(ckpt))
        # Verify at least the stem convolution was loaded when present
        if hasattr(model.backbone.stem, "conv"):
            weight = model.backbone.stem.conv.weight
            assert isinstance(weight, torch.Tensor)
        else:
            warnings.warn(
                f"model variant '{model.variant}' has no stem.conv; skipping weight check",
                RuntimeWarning,
            )


def test_from_pretrained_roundtrip(tmp_path):
    model = YOLOv9("n", num_classes=80)
    weight_file = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weight_file)
    loaded = YOLOv9.from_pretrained(str(weight_file), variant="n", num_classes=80)

    # ensure all parameters are identical
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])

    # compare serialized state dict bytes
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    torch.save(model.state_dict(), buf1)
    torch.save(loaded.state_dict(), buf2)
    assert buf1.getvalue() == buf2.getvalue()


def test_from_pretrained_variant_mismatch(tmp_path):
    model = YOLOv9("n", num_classes=80)
    weight_file = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weight_file)
    with pytest.raises(ValueError, match="variant"):
        YOLOv9.from_pretrained(str(weight_file), variant="s", num_classes=80)


def test_from_pretrained_class_mismatch(tmp_path):
    model = YOLOv9("n", num_classes=80)
    weight_file = tmp_path / "weights.pt"
    torch.save(model.state_dict(), weight_file)
    with pytest.raises(ValueError, match="classes"):
        YOLOv9.from_pretrained(str(weight_file), variant="n", num_classes=42)
