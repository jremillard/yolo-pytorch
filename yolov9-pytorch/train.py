"""Simple training script for YOLOv9 using COCO annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch import optim
from torch.utils.data import DataLoader

from models.yolov9 import YOLOv9
from utils.losses import YOLOLoss
from data import COCODataset, collate_fn


@torch.no_grad()
def build_targets(
    targets: List[Dict[str, torch.Tensor]],
    preds: List[torch.Tensor],
    num_classes: int,
    img_size: int,
) -> Dict[str, List[torch.Tensor]]:
    """Map ground truth boxes to model output grids."""
    strides = [img_size // p.shape[2] for p in preds]
    device = preds[0].device
    batch_size = len(targets)
    cls_t, box_t, mask_t = [], [], []
    for p, s in zip(preds, strides):
        _, _, h, w = p.shape
        cls_t.append(torch.zeros(batch_size, num_classes, h, w, device=device))
        box_t.append(torch.zeros(batch_size, 4, h, w, device=device))
        mask_t.append(torch.zeros(batch_size, 1, h, w, device=device))
    for b, t in enumerate(targets):
        boxes = t["boxes"] / img_size
        labels = t["labels"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        for box, cls, area in zip(boxes, labels, areas):
            if area < 0.02:
                idx = 0
            elif area < 0.15:
                idx = 1
            else:
                idx = 2
            h, w = cls_t[idx].shape[2:]
            cxcy = (box[:2] + box[2:]) / 2
            ij = (cxcy * torch.tensor([w, h], device=device)).long().clamp(0, h - 1)
            j, i = ij[1].item(), ij[0].item()
            mask_t[idx][b, 0, j, i] = 1
            box_t[idx][b, :, j, i] = box
            cls_t[idx][b, cls, j, i] = 1
    return {"cls": cls_t, "box": box_t, "mask": mask_t}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv9(args.variant, num_classes=args.num_classes).to(device)
    dataset = COCODataset(args.images, args.annotations, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    criterion = YOLOLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.train()
    for epoch in range(args.epochs):
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            t = build_targets(targets, preds, args.num_classes, args.img_size)
            loss = criterion(preds, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}: loss={loss.item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv9 on a COCO style dataset")
    parser.add_argument("--images", type=Path, required=True, help="Directory with training images")
    parser.add_argument("--annotations", type=Path, required=True, help="Path to COCO annotation JSON file")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--variant", type=str, default="n")
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
