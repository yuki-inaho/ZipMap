#!/usr/bin/env python3
"""Infer ZipMap depth/short-window poses on RTX 2070 with bounded VRAM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_zipmap_streaming_sequence import (
    load_and_preprocess_images,
    load_model,
    pose_encoding_to_extri_intri,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=280)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.image_size < 128 or args.image_size % 14:
        parser.error(
            "batch size must be positive; image size >=128 and divisible by 14"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    frames = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise FileNotFoundError("no input frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = load_model(
        args.checkpoint, False, torch.device("cuda"), dtype=torch.float16
    )
    for head_name in ("camera_mlp_head", "depth_head", "local_point_head"):
        head = getattr(model, head_name, None)
        if head is not None:
            head.float()
    loaded = time.perf_counter()
    rows = []
    for offset in range(0, len(frames), args.batch_size):
        chunk = frames[offset : offset + args.batch_size]
        images = load_and_preprocess_images(
            [str(path) for path in chunk], target_size=args.image_size, mode="pad"
        ).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            predictions = model(images, window_size=1)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        depth = predictions["depth"][0].float().cpu().numpy().squeeze(-1)
        confidence = predictions["depth_conf"][0].float().cpu().numpy()
        poses = extrinsics[0].float().cpu().numpy()
        k_matrices = intrinsics[0].float().cpu().numpy()
        for index, path in enumerate(chunk):
            np.savez_compressed(
                args.output_dir / f"{path.stem}.npz",
                depth=depth[index],
                depth_conf=confidence[index],
                extrinsics_chunk_w2c=poses[index],
                intrinsics=k_matrices[index],
                chunk_start_frame=chunk[0].name,
            )
            rows.append({"image": path.name, "chunk_start_frame": chunk[0].name})
        print(f"{offset + len(chunk)}/{len(frames)} frames", flush=True)
        del images, predictions, extrinsics, intrinsics
        torch.cuda.empty_cache()
    report = {
        "model": "ZipMap Streaming",
        "checkpoint": str(args.checkpoint.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "weight_dtype": "FP16 backbone, FP32 prediction heads",
        "frame_count": len(frames),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "preprocess": "portrait-preserving pad",
        "pose_scope": "each chunk has an independent TTT state and coordinate frame",
        "depth_units": "model units; calibrate to COLMAP with sparse 3D anchors",
        "load_seconds": loaded - start,
        "total_seconds": time.perf_counter() - start,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "frames": rows,
    }
    (args.output_dir / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}))


if __name__ == "__main__":
    main()
