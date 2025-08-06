"""Backbone modules for YOLO PyTorch project.

This module implements a minimal CSPDarknet backbone enhanced with GELAN
(Generalized Efficient Layer Aggregation Network) blocks.  The
implementation is intentionally lightweight but follows the high level
architecture described in the YOLOv9 paper: a Darknet style network with
Cross Stage Partial (CSP) connections and GELAN blocks.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None) -> int:
    """Compute padding size automatically."""
    return k // 2 if p is None else p


class Conv(nn.Module):
    """A standard convolution layer: Conv2d -> BatchNorm -> SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.act(self.bn(self.conv(x)))


class GELANBlock(nn.Module):
    """GELAN block consisting of two convolutional branches.

    Each branch performs a 1x1 reduction followed by a 3x3 convolution.
    The outputs of the branches are aggregated together with the input and
    passed through an activation function, mimicking the behaviour of the
    Generalized Efficient Layer Aggregation Network described in YOLOv9.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 1)
        self.branch1 = nn.Sequential(Conv(channels, hidden, 1), Conv(hidden, channels, 3))
        self.branch2 = nn.Sequential(Conv(channels, hidden, 1), Conv(hidden, channels, 3))
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        return self.act(self.bn(x + y1 + y2))


class GELAN(nn.Module):
    """Stack of :class:`GELANBlock` layers."""

    def __init__(self, channels: int, n: int = 1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[GELANBlock(channels) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.blocks(x)


class CSPDarknetGELAN(nn.Module):
    """CSPDarknet backbone with GELAN blocks.

    The network returns three feature maps corresponding to strides 8, 16
    and 32 relative to the input resolution.  It is purposely simplified
    but captures the essence of the original YOLOv9 backbone.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = Conv(3, c, k=3, s=2)
        self.stage1 = nn.Sequential(Conv(c, c * 2, k=3, s=2), GELAN(c * 2))
        self.stage2 = nn.Sequential(Conv(c * 2, c * 4, k=3, s=2), GELAN(c * 4))
        self.stage3 = nn.Sequential(Conv(c * 4, c * 8, k=3, s=2), GELAN(c * 8))
        self.stage4 = nn.Sequential(Conv(c * 8, c * 16, k=3, s=2), GELAN(c * 16))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale feature maps.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            list[torch.Tensor]: Feature maps at strides 8, 16 and 32.
        """

        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]


__all__ = ["CSPDarknetGELAN", "GELAN", "GELANBlock", "Conv"]
