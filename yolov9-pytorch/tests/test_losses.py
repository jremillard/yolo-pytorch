import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "yolov9-pytorch"))

from utils.losses import FocalLoss, IoULoss  # noqa: E402


def test_focal_loss_zero_when_correct():
    loss_fn = FocalLoss()
    pred = torch.tensor([10.0, -10.0])
    target = torch.tensor([1.0, 0.0])
    loss = loss_fn(pred, target)
    assert loss.item() < 1e-4


def test_iou_loss_basic():
    loss_fn = IoULoss()
    box1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    box2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert torch.isclose(loss_fn(box1, box2), torch.tensor(0.0), atol=1e-6).item()
    box3 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    box4 = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
    val = loss_fn(box3, box4).item()
    assert 0 < val < 1
