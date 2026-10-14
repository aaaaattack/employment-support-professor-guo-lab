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

## 当前基线验证

A6000 容器中使用 ModelScope 的 `AI-ModelScope/YOLO11/yolo11n.pt` 完成了固定输入 ONNX 导出：

```bash
python edge_deploy/export/export_onnx.py \
  --weights edge_deploy/models/pretrained/yolo11n.pt \
  --imgsz 640 --batch 1 --device 0

python edge_deploy/export/validate_onnx.py \
  --onnx edge_deploy/artifacts/onnx/yolo11n_b1_640_fp32.onnx \
  --run-runtime
```

验证结果：ONNX opset 17，输入 `images=[1,3,640,640]`，输出 `output0=[1,84,8400]`；CPU ONNX Runtime 执行成功且输出无 NaN/Inf。当前容器未安装 TensorRT Python 模块，因此 TensorRT engine 编译需在具备 TensorRT 的环境中进行。
