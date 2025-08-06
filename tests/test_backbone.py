import sys
from pathlib import Path

import torch

# Ensure the project root is on the import path for the models package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.backbone import (
    autopad,
    Conv,
    GELANBlock,
    GELAN,
    CSPDarknetGELAN,
)


def test_autopad():
    assert autopad(3) == 1
    assert autopad(5) == 2
    assert autopad(3, p=0) == 0


def test_conv_output_shape():
    conv = Conv(3, 16, k=3, s=2)
    x = torch.randn(1, 3, 32, 32)
    y = conv(x)
    assert y.shape == (1, 16, 16, 16)


def test_gelan_block_shape_and_grad():
    block = GELANBlock(32)
    x = torch.randn(2, 32, 64, 64, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None


def test_gelan_stack():
    gelan = GELAN(16, n=3)
    x = torch.randn(1, 16, 32, 32)
    y = gelan(x)
    assert y.shape == x.shape


def test_backbone_output_shapes():
    model = CSPDarknetGELAN(base_channels=32)
    x = torch.randn(1, 3, 256, 256)
    outputs = model(x)
    assert len(outputs) == 3
    assert outputs[0].shape == (1, 32 * 4, 32, 32)
    assert outputs[1].shape == (1, 32 * 8, 16, 16)
    assert outputs[2].shape == (1, 32 * 16, 8, 8)


def test_backbone_backward():
    model = CSPDarknetGELAN(base_channels=16)
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    outputs = model(x)
    loss = sum(o.mean() for o in outputs)
    loss.backward()
    assert x.grad is not None
