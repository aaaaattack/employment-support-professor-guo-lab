#!/usr/bin/env python3
"""Export an Ultralytics detector to a static ONNX graph."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "models" / "pretrained" / "yolo11n.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "onnx",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        dynamic=False,
        simplify=False,
        opset=args.opset,
        nms=False,
        verbose=True,
    )
    exported = Path(exported)
    target = args.output_dir / (
        f"{Path(args.weights).stem}_b{args.batch}_{args.imgsz}_"
        f"{'fp16' if args.half else 'fp32'}.onnx"
    )
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    print(f"exported={exported}")
    print(f"artifact={target}")


if __name__ == "__main__":
    main()
