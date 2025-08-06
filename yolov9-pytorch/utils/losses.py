"""Loss functions for YOLOv9 training."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Boxes are expected in ``(x1, y1, x2, y2)`` format.
    """
    # Intersection
    tl = torch.max(box1[:, None, :2], box2[:, :2])
    br = torch.min(box1[:, None, 2:], box2[:, 2:])
    inter = (br - tl).clamp(min=0).prod(dim=2)
    # Areas
    area1 = (box1[:, 2:] - box1[:, :2]).prod(dim=1)
    area2 = (box2[:, 2:] - box2[:, :2]).prod(dim=1)
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


class IoULoss(nn.Module):
    """IoU loss returning ``1 - IoU``."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        iou = box_iou(pred, target)
        loss = 1.0 - iou
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Binary focal loss with logits."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
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
    """Multi-label classification loss using BCE with logits."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.loss(pred, target)


class YOLOLoss(nn.Module):
    """Composite loss combining classification focal loss and IoU loss."""

    def __init__(self, num_classes: int, box_weight: float = 1.0, cls_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.focal = FocalLoss()
        self.iou = IoULoss()

    def forward(self, preds: List[torch.Tensor], targets: dict) -> torch.Tensor:  # noqa: D401
        total = torch.tensor(0.0, device=preds[0].device)
        for p, t_cls, t_box, t_mask in zip(preds, targets["cls"], targets["box"], targets["mask"]):
            box_pred = p[:, :4]
            cls_pred = p[:, 4:]
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
