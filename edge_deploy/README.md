# Ultralytics Edge Deployment

This directory contains the reproducible export, quantization, compilation, runtime, precision-debug, and benchmark workflow for NVIDIA RTX/A6000 and Ascend AIDK200. The upstream Ultralytics source is kept unchanged.

## Initial scope

- YOLO model export: PyTorch -> ONNX
- FP32/FP16/INT8 precision comparison
- TensorRT backend for NVIDIA GPUs
- ATC/OM/ACL backend for Ascend
- Layer-wise golden dumps and latency reports

## Directory layout

- `export/`: ONNX export and validation
- `quantization/`: calibration and quantization
- `nvidia/`: TensorRT build/runtime/benchmark
- `ascend/`: ATC and ACL build/runtime/benchmark
- `debug/`: native/ONNX/runtime tensor comparison
- `benchmark/`: end-to-end and model-only measurements
- `artifacts/`: generated files; do not commit large binaries
- `reports/`: reviewable accuracy and performance reports

The first implementation target is YOLO with batch size 1 and a static input shape. U-Net and ViT will reuse the same pipeline after the YOLO baseline is validated.
