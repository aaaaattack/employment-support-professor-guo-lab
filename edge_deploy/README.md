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

## 下一步：TensorRT 部署流程（按顺序执行）

### 1. 下载模型

```bash
modelscope download \
  --model AI-ModelScope/YOLO11 \
  --local_dir edge_deploy/models/pretrained \
  yolo11n.pt
```

### 2. 导出固定输入 ONNX

```bash
python edge_deploy/export/export_onnx.py \
  --weights edge_deploy/models/pretrained/yolo11n.pt \
  --imgsz 640 --batch 1 --device 0 \
  --output-dir edge_deploy/artifacts/onnx
```

默认产物：`edge_deploy/artifacts/onnx/yolo11n_b1_640_fp32.onnx`。

### 3. 校验 ONNX 图和 CPU 冒烟推理

```bash
python edge_deploy/export/validate_onnx.py \
  --onnx edge_deploy/artifacts/onnx/yolo11n_b1_640_fp32.onnx \
  --run-runtime
```

### 4. 在 TensorRT 环境构建 FP16 Engine

当前 A6000 容器未安装 TensorRT，因此下面命令需要切换到带 TensorRT 的 NVIDIA 容器或机器执行：

```bash
python edge_deploy/nvidia/build_engine.py \
  --onnx edge_deploy/artifacts/onnx/yolo11n_b1_640_fp32.onnx \
  --engine edge_deploy/artifacts/tensorrt/yolo11n_b1_640_fp16.engine \
  --workspace-gib 4 --fp16
```

执行前确认：

```bash
command -v trtexec
python -c "import tensorrt as tr; print(tr.__version__)"
```

### 5. ONNX 基线性能

当前容器只有 CPUExecutionProvider，可用于记录基线但不能代表 GPU/TensorRT 性能：

```bash
python edge_deploy/benchmark/benchmark_onnx.py \
  --onnx edge_deploy/artifacts/onnx/yolo11n_b1_640_fp32.onnx \
  --warmup 10 --iters 100
```

已完成的 30 次冒烟结果：CPU ONNX Runtime，mean `41.112 ms`，p50 `36.558 ms`，p90 `48.266 ms`。该结果仅用于图正确性和回归比较，不作为最终边缘设备指标。

### 6. 后续精度/性能对比

TensorRT 环境就绪后，按以下顺序补测并记录到 `reports/`：

1. 固定同一份输入图像和预处理结果；
2. 对比 PyTorch、ONNX Runtime、TensorRT 的原始输出余弦相似度、MAE、最大误差；
3. 预热 10 次后测量 100 次，报告 mean/p50/p90/p99；
4. 最后再接入 NMS 和端到端后处理，单独报告端到端耗时。
