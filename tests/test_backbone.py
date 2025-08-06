import torch

from models.backbone import CSPDarknetGELAN


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
