"""Loss functions used during training of the simplified YOLOv9 model.

The aim is to keep the implementation concise while remaining useful for real
training scenarios.  The losses defined here are intentionally lightweight, but
they are sufficient to fine tune models initialised from official weights and
to support end-to-end inference.  Each function is accompanied by detailed
comments that favour clarity over absolute feature parity with the reference
code base.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of axis-aligned boxes.

    Args:
        box1: Tensor of shape ``(N, 4)`` in ``(x1, y1, x2, y2)`` format.
        box2: Tensor of shape ``(M, 4)`` in the same format.

    Returns:
        ``(N, M)`` tensor containing the IoU for every pair of boxes.
    """

    # Intersection top left & bottom right corners.
    tl = torch.max(box1[:, None, :2], box2[:, :2])
    br = torch.min(box1[:, None, 2:], box2[:, 2:])
    inter = (br - tl).clamp(min=0).prod(dim=2)

    # Areas of individual boxes and union area.
    area1 = (box1[:, 2:] - box1[:, :2]).prod(dim=1)
    area2 = (box2[:, 2:] - box2[:, :2]).prod(dim=1)
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


class IoULoss(nn.Module):
    """Loss based on the Intersection over Union metric.

    The module expects ``pred`` and ``target`` tensors containing bounding
    boxes in ``(x1, y1, x2, y2)`` format.  The loss is defined as ``1 - IoU`` so
    that perfect overlap yields zero loss.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute the IoU loss between ``pred`` and ``target`` boxes."""

        iou = box_iou(pred, target)
        loss = 1.0 - iou
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Binary focal loss with logits.

    Focal loss down-weights easy examples and focuses the training on hard
    negatives.  This implementation operates directly on logits and therefore
    wraps :class:`torch.nn.BCEWithLogitsLoss`.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute focal loss given raw predictions and targets."""

        bce = self.bce(pred, target)
        prob = torch.sigmoid(pred)
        pt = target * prob + (1 - target) * (1 - prob)
        alpha = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss = alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassificationLoss(nn.Module):
    """Multi-label classification loss implemented with BCE with logits."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return standard binary cross-entropy loss for multi-label targets."""

        return self.loss(pred, target)


class YOLOLoss(nn.Module):
    """Composite detection loss combining classification and box regression.

    The loss expects predictions for multiple feature levels and a dictionary of
    target tensors with keys ``"cls"``, ``"box"`` and ``"mask"``.  The mask
    indicates which spatial locations contain objects and both the box and
    classification terms are evaluated only at those positive locations.  This
    keeps the implementation concise while remaining suitable for fine tuning
    the simplified model.  For simplicity only IoU loss and focal classification
    loss are used.
    """

    def __init__(self, num_classes: int, box_weight: float = 1.0, cls_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.focal = FocalLoss()
        self.iou = IoULoss()

    def forward(self, preds: List[torch.Tensor], targets: dict) -> torch.Tensor:  # noqa: D401
        """Compute the aggregate loss over all feature levels."""

        total = torch.tensor(0.0, device=preds[0].device)
        for p, t_cls, t_box, t_mask in zip(preds, targets["cls"], targets["box"], targets["mask"]):
            # Split predicted tensor into box and classification parts.
            box_pred = p[:, :4]
            cls_pred = p[:, 4:]

            # Only compute loss for positive locations as indicated by ``mask``.
            mask = t_mask.bool().squeeze(1)
            if mask.any():
                box_loss = self.iou(box_pred.permute(0, 2, 3, 1)[mask], t_box.permute(0, 2, 3, 1)[mask])
                cls_loss = self.focal(cls_pred.permute(0, 2, 3, 1)[mask], t_cls.permute(0, 2, 3, 1)[mask])
                total = total + self.box_weight * box_loss + self.cls_weight * cls_loss
        return total


__all__ = [
    "FocalLoss",
    "IoULoss",
    "ClassificationLoss",
    "YOLOLoss",
    "box_iou",
]
