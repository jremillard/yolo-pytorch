# Plan for Porting YOLO to a Clean PyTorch Implementation

This document outlines the steps needed to reimplement YOLOv9 in pure PyTorch without relying on Ultralytics or other external "ultra analysis" dependencies. The plan assumes access to the original repository, published paper, pretrained weights, and a benchmark image set for validation.

## 1. Repository Study and Baseline
- Inspect the existing `yolov9-pytorch` directory to understand the current architecture, utilities, and tests.
- Review commit history to track model, loss, and training script evolution. Identify pieces that still mirror the original repo and note areas requiring a clean-room rewrite.
- Document current feature coverage for YOLOv9 variants: **t**, **s**, **m**, **c**, and **e**.

## 2. Environment and Dependency Isolation
- Create a minimal `requirements` list limited to PyTorch, Torchvision, NumPy, and essential utilities (OpenCV, matplotlib, PyYAML).
- Remove any implicit Ultralytics references such as custom data loaders or model helper functions. Replace them with equivalent utilities coded from scratch or using standard libraries.

## 3. Model Architecture
1. **Backbone (CSPDarknet with GELAN blocks)**
   - Implement convolutional blocks, bottlenecks, and GELAN modules directly in PyTorch.
   - Parameterize width and depth multipliers to scale across all variant sizes.
2. **Neck**
   - Implement feature pyramid and path aggregation layers.
   - Ensure dynamic shape handling for different input resolutions.
3. **Detection Head**
   - Define anchor-free detection head producing classification, objectness, and box regression outputs.
   - Configure per-variant channel counts and strides.

## 4. Weight Conversion
- Write a script to load weights from the original repo and map them to the clean PyTorch model's parameter names.
- Validate mapping by running a forward pass on a known image and confirming numerical parity with the original implementation.

## 5. Data Pipeline
- Implement dataset classes that read YOLO-format annotations and images.
- Recreate augmentations (mosaic, mixup, color jitter, HSV shift) using OpenCV or torchvision transforms.
- Ensure deterministic behavior through configurable random seeds.

## 6. Training Loop
- Build a trainer using pure PyTorch constructs:
  - SGD or AdamW optimizer.
  - Cosine or one-cycle learning rate scheduler.
  - Mixed precision via `torch.cuda.amp`.
- Implement losses: IoU-based box loss, objectness loss, and classification loss with optional label smoothing and weighting.
- Support model warmup, gradient accumulation, and EMA of model weights.

## 7. Inference and Export
- Implement non-max suppression using `torchvision.ops.nms` to avoid third‑party implementations.
- Provide scripts for image inference, batch inference, and ONNX export.
- Include optional TensorRT/ONNXRuntime hooks while keeping the core code free from those dependencies.

## 8. Evaluation and Benchmarking
- Use the provided benchmark image set to compute mAP and latency.
- Compare results with those from the original repository to ensure fidelity.
- Record benchmarks for each variant size and document hardware/software configuration.

## 9. Testing Strategy
- Create unit tests for each module: backbone blocks, neck, head, loss functions, and dataset loader.
- Include integration tests that load pretrained weights and perform a single forward pass.
- Maintain CI scripts executing `pytest -q` to guarantee reliability.

## 10. Documentation and Maintenance
- Update the README to describe the clean-room implementation and usage instructions.
- Provide architecture diagrams and configuration tables for all variants.
- Add contribution guidelines emphasizing separation from the original codebase.

This plan establishes a path to a standalone PyTorch YOLOv9 implementation that retains compatibility with existing weights and benchmarks while avoiding dependencies on Ultralytics or other proprietary helper libraries.
