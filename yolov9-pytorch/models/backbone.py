"""Backbone modules for the custom YOLOv9 implementation.

The aim of this file is to provide a small, easy to read recreation of the
backbone described in the `YOLOv9` paper.  It borrows the overall layout of a
`CSPDarknet` network and enhances the core blocks with *GELAN* (Generalized
Efficient Layer Aggregation Network) style aggregation.  Only a subset of the
original architecture is implemented but the main ideas are preserved:

* use simple Conv→BN→SiLU blocks throughout
* aggregate features with GELAN blocks
* expose feature maps at strides 8, 16 and 32 for the detection head

The code is intentionally compact so the comments below explain the reasoning
behind each step in detail.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None) -> int:
    """Compute padding size automatically.

    When no explicit padding ``p`` is provided we default to ``k // 2``.  This
    emulates the behaviour of ``padding='same'`` which keeps spatial
    dimensions constant for odd ``k``.  The helper mirrors the logic used in
    the reference implementation and keeps layer definitions concise.
    """

    return k // 2 if p is None else p


class Conv(nn.Module):
    """Convolutional block used throughout the model.

    The block performs a 2D convolution followed by batch normalisation and a
    ``SiLU`` activation.  This mirrors the conv blocks employed by the official
    implementation and avoids repeating the pattern in every module.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        k: Convolution kernel size.
        s: Stride of the convolution.
        p: Optional explicit padding.  When ``None`` the :func:`autopad`
            helper is used to emulate ``padding='same'`` behaviour.
        groups: Convolution groups for grouped convolutions.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, groups: int = 1) -> None:
        super().__init__()
        # ``bias=False`` is standard practice when BatchNorm is used as it
        # already provides a learnable bias term.
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply the Conv→BN→SiLU sequence to the input tensor."""

        return self.act(self.bn(self.conv(x)))


class GELANBlock(nn.Module):
    """Minimal GELAN block with two aggregation branches.

    The block reproduces the aggregation principle of GELAN/E-ELAN from the
    paper: the input tensor is processed by two parallel branches.  Each branch
    first reduces the number of channels with a ``1×1`` convolution and then
    expands features with a ``3×3`` convolution.  Their outputs are added to the
    original input (residual connection) and passed through a final normalisation
    and activation step.  The design promotes feature reuse while keeping the
    number of parameters small.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # The hidden dimension controls the bottleneck size.  A ``max`` guard is
        # used to avoid creating zero-width layers when ``channels`` is tiny
        # (e.g. in the ``t`` variant).
        hidden = max(channels // 2, 1)
        self.branch1 = nn.Sequential(Conv(channels, hidden, 1), Conv(hidden, channels, 3))
        self.branch2 = nn.Sequential(Conv(channels, hidden, 1), Conv(hidden, channels, 3))
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return aggregated features from both branches and the shortcut."""

        # Process the input through the two branches.
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        # Combine branch outputs with the original input and apply BN + SiLU.
        return self.act(self.bn(x + y1 + y2))


class GELAN(nn.Module):
    """A simple container stacking multiple :class:`GELANBlock` objects."""

    def __init__(self, channels: int, n: int = 1) -> None:
        super().__init__()
        # ``n`` controls the depth of the block stack.  In the official models
        # the number of repetitions grows with the model scale; here we mimic
        # that behaviour via ``depth_multiplier`` in :class:`CSPDarknetGELAN`.
        self.blocks = nn.Sequential(*[GELANBlock(channels) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply all stacked GELAN blocks sequentially."""

        return self.blocks(x)


class CSPDarknetGELAN(nn.Module):
    """CSPDarknet backbone augmented with GELAN blocks.

    The original YOLOv9 backbone uses a combination of CSP connections and
    GELAN/E-ELAN blocks to efficiently scale depth and width.  This condensed
    version keeps the same spirit: four sequential stages progressively reduce
    spatial resolution while increasing channel capacity.  The network exposes
    the outputs of the last three stages which correspond to strides ``8``,
    ``16`` and ``32`` relative to the input and are consumed by the detection
    head.

    Args:
        base_channels: Number of channels in the first stage.  Larger values
            yield wider networks.
        depth_multiplier: Scales the number of GELAN blocks per stage, allowing
            the creation of tiny through large variants using the same code.
    """

    def __init__(self, base_channels: int = 32, depth_multiplier: float = 1.0) -> None:
        super().__init__()
        c = base_channels

        def d(n: int) -> int:
            """Round ``n`` based on ``depth_multiplier`` ensuring at least 1."""

            return max(int(round(n * depth_multiplier)), 1)

        # Stem: initial downsampling from RGB input.
        self.stem = Conv(3, c, k=3, s=2)

        # Each stage halves the spatial dimensions (stride 2) and expands the
        # channel width.  A stack of GELAN blocks then refines the features.
        self.stage1 = nn.Sequential(Conv(c, c * 2, k=3, s=2), GELAN(c * 2, n=d(1)))
        self.stage2 = nn.Sequential(Conv(c * 2, c * 4, k=3, s=2), GELAN(c * 4, n=d(2)))
        self.stage3 = nn.Sequential(Conv(c * 4, c * 8, k=3, s=2), GELAN(c * 8, n=d(3)))
        self.stage4 = nn.Sequential(Conv(c * 8, c * 16, k=3, s=2), GELAN(c * 16, n=d(1)))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning multi-scale feature maps.

        Args:
            x: Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            list[torch.Tensor]: Feature maps at strides 8, 16 and 32.
        """

        # ``p3``/``p4``/``p5`` naming follows the FPN convention.
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]


__all__ = ["CSPDarknetGELAN", "GELAN", "GELANBlock", "Conv"]
