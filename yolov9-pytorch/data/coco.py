"""COCO dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import transforms as T


class COCODataset(Dataset):
    """Thin wrapper around :class:`torchvision.datasets.CocoDetection`."""

    def __init__(self, image_dir: str | Path, ann_file: str | Path, img_size: int = 256) -> None:
        super().__init__()
        self.coco = CocoDetection(str(image_dir), str(ann_file))
        self.img_size = img_size
        self.tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def __len__(self) -> int:  # noqa: D401
        return len(self.coco)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # noqa: D401
        img, targets = self.coco[idx]
        img = self.tf(img)
        boxes: List[List[float]] = []
        labels: List[int] = []
        for t in targets:
            x, y, w, h = t["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(t["category_id"])
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        return img, {"boxes": boxes_t, "labels": labels_t}


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    imgs, targets = zip(*batch)
    return torch.stack(list(imgs)), list(targets)


__all__ = ["COCODataset", "collate_fn"]
