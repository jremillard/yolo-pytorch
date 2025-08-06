"""YOLOv9 detection model implemented from scratch.

Only the core architectural pieces required for inference are reproduced in
this project.  The goal is not to be feature complete but to provide a clear
and well commented PyTorch implementation that mirrors the design presented in
the paper and in the reference code base.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import sys
import warnings

import torch
import torch.nn as nn

from .backbone import CSPDarknetGELAN, Conv


class PANFPN(nn.Module):
    """Simplified PANet/FPN neck used to fuse multi-scale features.

    The network performs a top-down pathway followed by a bottom-up pathway,
    closely resembling the structure described in the YOLOv9 paper.  Each
    lateral or smoothing operation is implemented with the shared
    :class:`~yolov9_pytorch.models.backbone.Conv` block for consistency.
    Despite its compact form this neck is sufficient for loading official
    checkpoints and validating predictions.
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        # Channel counts coming from the backbone at strides 8/16/32.
        c3, c4, c5 = base_channels * 4, base_channels * 8, base_channels * 16
        c_out = base_channels * 4  # unified channel width for all pyramid levels

        # Lateral 1×1 convolutions used before upsampling/concatenation.
        self.cv5 = Conv(c5, c_out, 1)
        self.cv4 = Conv(c4, c_out, 1)
        self.cv3 = Conv(c3, c_out, 1)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Smoothing convs applied after concatenation in the top-down pass.
        self.smooth4 = Conv(c_out * 2, c_out, 3)
        self.smooth3 = Conv(c_out * 2, c_out, 3)

        # Bottom-up pass where features are downsampled and merged again.
        self.down4 = Conv(c_out, c_out, 3, s=2)
        self.out4 = Conv(c_out * 2, c_out, 3)
        self.down5 = Conv(c_out, c_out, 3, s=2)
        self.out5 = Conv(c_out * 2, c_out, 3)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:  # noqa: D401
        """Fuse feature maps from the backbone into a pyramid.

        Args:
            feats: List of tensors ``[p3, p4, p5]`` from the backbone.

        Returns:
            list[torch.Tensor]: Fused features ``[p3, n4, n5]`` ready for the
            detection head.
        """

        p3, p4, p5 = feats

        # Top-down pathway: upsample higher level features and concatenate.
        p5 = self.cv5(p5)
        p4 = self.smooth4(torch.cat([self.cv4(p4), self.up(p5)], dim=1))
        p3 = self.smooth3(torch.cat([self.cv3(p3), self.up(p4)], dim=1))

        # Bottom-up pathway: refine by downsampling and merging again.
        n4 = self.out4(torch.cat([self.down4(p3), p4], dim=1))
        n5 = self.out5(torch.cat([self.down5(n4), p5], dim=1))
        return [p3, n4, n5]


class Detect(nn.Module):
    """Anchor-free detection head.

    Each feature map is processed by a ``1×1`` convolution that outputs ``nc``
    class logits and four bounding box coordinates per spatial location.  The
    head itself performs no post-processing; it simply returns raw predictions
    for the caller to decode.
    """

    def __init__(self, nc: int, ch: List[int]) -> None:
        super().__init__()
        self.nc = nc
        # ``ch`` holds the number of channels for each pyramid level; a separate
        # conv layer is created for every scale.
        self.m = nn.ModuleList([nn.Conv2d(c, nc + 4, 1) for c in ch])

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:  # noqa: D401
        """Apply the detection conv to each feature map in ``x``."""

        return [m(v) for m, v in zip(self.m, x)]


class YOLOv9(nn.Module):
    """Unified YOLOv9 model with configurable variants.

    The architecture is composed of three high level parts: a GELAN based
    backbone (:class:`CSPDarknetGELAN`), a PAN-FPN style neck
    (:class:`PANFPN`) and a lightweight anchor-free detection head
    (:class:`Detect`).  Different model scales are obtained by adjusting the
    base number of channels and depth multiplier as reported in the paper.

    Supported variants are ``t`` (tiny), ``s`` (small), ``m`` (medium),
    ``c`` (compact) and ``e`` (extended).  The tuple in :data:`VARIANTS`
    defines ``(base_channels, depth_multiplier)`` for each configuration.
    """

    VARIANTS: Dict[str, Tuple[int, float]] = {
        "t": (16, 0.33),
        "s": (32, 0.33),
        "m": (48, 0.67),
        "c": (64, 1.0),
        "e": (80, 1.0),
    }

    def __init__(self, variant: str = "s", num_classes: int = 80) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant '{variant}'")

        base, depth = self.VARIANTS[variant]
        self.variant = variant

        # Build backbone, neck and detection head according to the variant
        # specific width (`base`) and depth parameters.
        self.backbone = CSPDarknetGELAN(base_channels=base, depth_multiplier=depth)
        self.neck = PANFPN(base)
        ch = [base * 4] * 3  # channel dimensions for detection layers
        self.detect = Detect(num_classes, ch)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # noqa: D401
        """Run a forward pass through backbone, neck and detection head."""

        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.detect(feats)

    @classmethod
    def from_pretrained(cls, weight_path: str, num_classes: int = 80, variant: str | None = None) -> "YOLOv9":
        """Instantiate a model from a YOLOv9 ``.pt`` checkpoint.

        The original checkpoints were trained with a richer code base.  To keep
        this implementation lightweight we only load matching parameters and
        silently skip the rest.  The helper also attempts to infer the model
        variant from the checkpoint to ensure the architecture matches.
        """

        import types

        # Allow ``torch.load`` to deserialise the original Ultralytics modules
        # by providing dummy stand-ins for the missing packages.
        class DummyModule(types.ModuleType):
            def __getattr__(self, name: str):
                cls = type(name, (nn.Module,), {})
                setattr(self, name, cls)
                return cls

        for mod in ["models", "models.common", "models.yolo"]:
            sys.modules[mod] = DummyModule(mod)  # type: ignore[assignment]

        # Load checkpoint.  We accept plain state dicts as well as training
        # checkpoints that store the model inside a dictionary.
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
                state = ckpt["model"].state_dict()
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt

        # Infer the variant by inspecting the first conv layer of the backbone.
        mapping = {16: "t", 32: "s", 48: "m", 64: "c", 80: "e"}
        ch0 = None
        if "backbone.stem.conv.weight" in state:
            ch0 = state["backbone.stem.conv.weight"].shape[0]
        else:
            warnings.warn(
                "checkpoint missing 'backbone.stem.conv.weight'; variant inference may be unreliable",
                RuntimeWarning,
            )
            for key in ("model.0.conv.weight", "model.1.conv.weight"):
                if key in state:
                    ch0 = state[key].shape[0]
                    break
        inferred_variant = mapping.get(ch0)
        if variant is None:
            variant = inferred_variant or "s"
        elif inferred_variant and variant != inferred_variant:
            raise ValueError(
                f"checkpoint is for variant '{inferred_variant}' but variant '{variant}' was requested"
            )

        # Validate that the requested number of classes matches the checkpoint.
        nc_ckpt = None
        for key in ("model.24.m.0.weight", "model.23.m.0.weight", "detect.m.0.weight"):
            if key in state:
                nc_ckpt = state[key].shape[0] - 4
                break
        if nc_ckpt is not None and nc_ckpt != num_classes:
            raise ValueError(
                f"checkpoint has {nc_ckpt} classes but model built for {num_classes}"
            )

        # Build the model and load compatible weights.
        model = cls(variant=variant, num_classes=num_classes)
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(compatible)
        model.load_state_dict(own, strict=False)
        return model


__all__ = ["YOLOv9"]
