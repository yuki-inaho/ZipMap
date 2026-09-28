#!/usr/bin/env python3
"""Infer Omega depth/short-window poses for a long RGB sequence on one RTX 2070."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.image_size < 128 or args.image_size % 16:
        parser.error(
            "batch size must be positive; image size >=128 and divisible by 16"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
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
    model = VGGTOmega().eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model = model.cuda()
    loaded = time.perf_counter()
    autocast_dtype = (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )
    rows = []
    for offset in range(0, len(frames), args.batch_size):
        chunk = frames[offset : offset + args.batch_size]
        images = load_and_preprocess_images(
            [str(path) for path in chunk],
            mode="max_size",
            image_resolution=args.image_size,
        ).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
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
        "model": "VGGT-Omega-1B-512",
        "checkpoint": str(args.checkpoint.resolve()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "autocast_dtype": str(autocast_dtype),
        "frame_count": len(frames),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "pose_scope": "each chunk has an independent coordinate frame; do not concatenate chunk poses",
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
