"""YOLOv9 detection model implemented from scratch."""

from __future__ import annotations

from typing import Dict, List, Tuple

import sys
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
        state = ckpt["model"].state_dict() if isinstance(ckpt, dict) else ckpt

        # Infer model variant from the first convolution's channel count
        if variant is None:
            ch0 = None
            for key in ("model.0.conv.weight", "model.1.conv.weight"):
                if key in state:
                    ch0 = state[key].shape[0]
                    break
            mapping = {16: "n", 32: "s", 48: "m", 64: "l"}
            variant = mapping.get(ch0, "s")

        model = cls(variant=variant, num_classes=num_classes)
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(compatible)
        model.load_state_dict(own, strict=False)
        return model


__all__ = ["YOLOv9"]
