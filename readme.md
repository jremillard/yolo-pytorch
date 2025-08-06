# YOLOv9 Native PyTorch Implementation

A clean, native PyTorch implementation of YOLOv9 (You Only Look Once version 9) object detection model.

## Overview

This project provides a pure PyTorch implementation of YOLOv9, focusing on:
- Clean, readable code structure
- Native PyTorch operations (no external YOLO frameworks)
- Modular architecture for easy customization
- Training and inference capabilities
- Support for custom datasets

### Model Components

```
Orginal YOLO v9 code.
yolov9-org
├── Backbone (CSPDarknet + GELAN)
├── Neck (PAN-FPN with PGI)
├── Detection Head (Anchor-free)
└── Loss Functions (Focal + IoU + Classification)
```

## Project Structure

```
yolo-pytorch/
├── models/
├── utils/
├── data/
├── configs/
├── train.py                # Training script
├── inference.py            # Inference script
├── export.py               # Model export (ONNX, TorchScript)
└── requirements.txt        # Dependencies
```

## YOLOv9 Paper

The repository includes the LaTeX source for the YOLOv9 paper under
`arXiv-2402.13616v2/yolov9_arxiv.tex` for reference.

