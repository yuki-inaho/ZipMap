#!/usr/bin/env python3
"""Run one model in its own Python environment for the comparison pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["zipmap", "omega"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )[: args.max_frames]
    if not paths:
        raise FileNotFoundError(f"No images in {args.input_dir}")

    start = time.perf_counter()
    if args.model == "zipmap":
        from run_zipmap_streaming_sequence import (
            load_and_preprocess_images,
            load_model,
            pose_encoding_to_extri_intri,
        )

        model = load_model(args.checkpoint, False, torch.device("cuda"))
        images = load_and_preprocess_images([str(path) for path in paths]).cuda()
        decode = pose_encoding_to_extri_intri
    else:
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        model = VGGTOmega().eval()
        model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        model = model.cuda()
        images = load_and_preprocess_images([str(path) for path in paths]).cuda()
        decode = encoding_to_camera
    loaded = time.perf_counter()
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = (
            model(images, window_size=1) if args.model == "zipmap" else model(images)
        )
    torch.cuda.synchronize()
    inferred = time.perf_counter()

    extrinsics, intrinsics = decode(predictions["pose_enc"], images.shape[-2:])
    homogeneous = torch.eye(4, device="cuda").repeat(1, len(paths), 1, 1)
    homogeneous[:, :, :3] = extrinsics
    homogeneous = homogeneous @ torch.linalg.inv(homogeneous[:, :1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        frame_names=np.array([path.name for path in paths]),
        extrinsics_world_to_camera=homogeneous[0, :, :3].float().cpu().numpy(),
        intrinsics=intrinsics[0].float().cpu().numpy(),
        depth=predictions["depth"][0].float().cpu().numpy(),
        depth_conf=predictions["depth_conf"][0].float().cpu().numpy(),
    )
    np.save(
        args.output_dir / "rgb.npy",
        images.permute(0, 2, 3, 1).mul(255).round().byte().cpu().numpy(),
    )
    summary = {
        "model": args.model,
        "frame_count": len(paths),
        "image_shape": list(images.shape),
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "load_seconds": loaded - start,
        "inference_seconds": inferred - loaded,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (args.output_dir / "runtime.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
