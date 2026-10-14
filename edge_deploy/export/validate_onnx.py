"""Validate an exported ONNX graph and optionally execute one CPU inference."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

def _concrete_shape(shape):
    out = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        else:
            out.append(1)
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--run-runtime", action="store_true")
    args = parser.parse_args()
    model = onnx.load(str(args.onnx))
    onnx.checker.check_model(model)
    print(f"graph_ok={args.onnx}")
    print(f"ir_version={model.ir_version} opset={[x.version for x in model.opset_import]}")
    for value in model.graph.input:
        tensor = value.type.tensor_type
        shape = [d.dim_value if d.dim_value else (d.dim_param or "dynamic") for d in tensor.shape.dim]
        print(f"input name={value.name} elem_type={tensor.elem_type} shape={shape}")
    for value in model.graph.output:
        tensor = value.type.tensor_type
        shape = [d.dim_value if d.dim_value else (d.dim_param or "dynamic") for d in tensor.shape.dim]
        print(f"output name={value.name} elem_type={tensor.elem_type} shape={shape}")
    if not args.run_runtime:
        return
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    feeds = {}
    for meta in session.get_inputs():
        shape = _concrete_shape(meta.shape)
        dtype = np.float16 if "float16" in meta.type else np.float32
        feeds[meta.name] = np.random.rand(*shape).astype(dtype)
        print(f"runtime_input name={meta.name} shape={shape} dtype={feeds[meta.name].dtype}")
    outputs = session.run(None, feeds)
    for index, output in enumerate(outputs):
        arr = np.asarray(output)
        print(f"runtime_output[{index}] shape={arr.shape} dtype={arr.dtype} finite={np.isfinite(arr).all()}")

if __name__ == "__main__":
    main()
