#!/usr/bin/env python3
"""Benchmark an ONNX graph through ONNX Runtime."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=100)
    args = p.parse_args()
    sess = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    meta = sess.get_inputs()[0]
    shape = [int(x) if isinstance(x, int) and x > 0 else 1 for x in meta.shape]
    dtype = np.float16 if 'float16' in meta.type else np.float32
    x = np.random.rand(*shape).astype(dtype)
    for _ in range(args.warmup):
        sess.run(None, {meta.name: x})
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, {meta.name: x})
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    print(f'provider={sess.get_providers()} input={shape} dtype={dtype}')
    print(f'iters={len(a)} mean_ms={a.mean():.3f} p50_ms={np.percentile(a,50):.3f} p90_ms={np.percentile(a,90):.3f} min_ms={a.min():.3f} max_ms={a.max():.3f}')

if __name__ == '__main__':
    main()
