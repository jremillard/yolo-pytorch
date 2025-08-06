"""YOLOv9 detection model implemented from scratch."""

from __future__ import annotations

from typing import Dict, List, Tuple

import sys
import warnings

import torch
import torch.nn as nn

from .backbone import CSPDarknetGELAN, Conv


class PANFPN(nn.Module):
    """Simple PAN-style feature pyramid network.

    It performs a top-down and bottom-up pass using :class:`Conv` blocks.
    The implementation is intentionally lightweight but sufficient for
    loading pretrained weights and running basic inference tests.
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        c3, c4, c5 = base_channels * 4, base_channels * 8, base_channels * 16
        c_out = base_channels * 4
        self.cv5 = Conv(c5, c_out, 1)
        self.cv4 = Conv(c4, c_out, 1)
        self.cv3 = Conv(c3, c_out, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.smooth4 = Conv(c_out * 2, c_out, 3)
        self.smooth3 = Conv(c_out * 2, c_out, 3)
        self.down4 = Conv(c_out, c_out, 3, s=2)
        self.out4 = Conv(c_out * 2, c_out, 3)
        self.down5 = Conv(c_out, c_out, 3, s=2)
        self.out5 = Conv(c_out * 2, c_out, 3)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:  # noqa: D401
        p3, p4, p5 = feats
        p5 = self.cv5(p5)
        p4 = self.smooth4(torch.cat([self.cv4(p4), self.up(p5)], dim=1))
        p3 = self.smooth3(torch.cat([self.cv3(p3), self.up(p4)], dim=1))
        n4 = self.out4(torch.cat([self.down4(p3), p4], dim=1))
        n5 = self.out5(torch.cat([self.down5(n4), p5], dim=1))
        return [p3, n4, n5]


class Detect(nn.Module):
    """Detection head predicting bounding boxes and class scores."""

    def __init__(self, nc: int, ch: List[int]) -> None:
        super().__init__()
        self.nc = nc
        self.m = nn.ModuleList([nn.Conv2d(c, nc + 4, 1) for c in ch])

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:  # noqa: D401
        return [m(v) for m, v in zip(self.m, x)]


class YOLOv9(nn.Module):
    """Unified YOLOv9 model with configurable variants."""

    VARIANTS: Dict[str, Tuple[int, float]] = {
        "n": (16, 0.33),
        "s": (32, 0.33),
        "m": (48, 0.67),
        "l": (64, 1.0),
    }

    def __init__(self, variant: str = "s", num_classes: int = 80) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant '{variant}'")
        base, depth = self.VARIANTS[variant]
        self.variant = variant
        self.backbone = CSPDarknetGELAN(base_channels=base, depth_multiplier=depth)
        self.neck = PANFPN(base)
        ch = [base * 4] * 3
        self.detect = Detect(num_classes, ch)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # noqa: D401
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.detect(feats)

    @classmethod
    def from_pretrained(cls, weight_path: str, num_classes: int = 80, variant: str | None = None) -> "YOLOv9":
        """Create model and load weights from a YOLOv9 ``.pt`` checkpoint.

        The method performs a best-effort loading: weights that do not match
        the current architecture are skipped.  This allows loading official
        checkpoints even though this implementation is simplified.
        """

        import types

        # Allow ``torch.load`` to deserialise the original Ultralytics modules
        class DummyModule(types.ModuleType):
            def __getattr__(self, name: str):
                cls = type(name, (nn.Module,), {})
                setattr(self, name, cls)
                return cls

        for mod in ["models", "models.common", "models.yolo"]:
            sys.modules[mod] = DummyModule(mod)  # type: ignore[assignment]

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

        mapping = {16: "n", 32: "s", 48: "m", 64: "l"}
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

        nc_ckpt = None
        for key in ("model.24.m.0.weight", "model.23.m.0.weight", "detect.m.0.weight"):
            if key in state:
                nc_ckpt = state[key].shape[0] - 4
                break
        if nc_ckpt is not None and nc_ckpt != num_classes:
            raise ValueError(
                f"checkpoint has {nc_ckpt} classes but model built for {num_classes}"
            )

        model = cls(variant=variant, num_classes=num_classes)
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(compatible)
        model.load_state_dict(own, strict=False)
        return model


__all__ = ["YOLOv9"]
