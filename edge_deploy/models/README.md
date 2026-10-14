# Model assets

预训练权重通过 ModelScope 获取，避免依赖 GitHub Releases 的下载链路：

```bash
modelscope download \
  --model AI-ModelScope/YOLO11 \
  --local_dir edge_deploy/models/pretrained \
  yolo11n.pt
```

默认导出脚本会读取：

```text
edge_deploy/models/pretrained/yolo11n.pt
```

模型权重和导出的 ONNX 文件属于二进制产物，不提交到 Git；请在部署机器上按上述命令重新下载。
