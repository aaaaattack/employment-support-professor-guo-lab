#!/usr/bin/env python3
"""Build a TensorRT engine from a static ONNX graph."""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--workspace-gib', type=int, default=4)
    p.add_argument('--fp16', action='store_true')
    args = p.parse_args()
    try:
        import tensorrt as tr
    except ImportError as exc:
        raise SystemExit('TensorRT Python is unavailable; install TensorRT or run in a TensorRT container.') from exc
    logger = tr.Logger(tr.Logger.WARNING)
    builder = tr.Builder(logger)
    flags = 1 << int(tr.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = tr.OnnxParser(network, logger)
    data = args.onnx.read_bytes()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit('TensorRT ONNX parsing failed')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(tr.MemoryPoolType.WORKSPACE, args.workspace_gib * (1 << 30))
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(tr.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit('TensorRT engine build failed')
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    print(f'engine={args.engine} bytes={args.engine.stat().st_size}')

if __name__ == '__main__':
    main()
