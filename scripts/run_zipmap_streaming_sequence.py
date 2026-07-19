#!/usr/bin/env python3
"""Run the ZipMap Streaming model on a sorted RGB image sequence without Gradio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from zipmap.models.ZipMap_AR import ZipMap
from zipmap.utils.load_fn import load_and_preprocess_images
from zipmap.utils.pose_enc import pose_encoding_to_extri_intri


STREAMING_CONFIG = {
    "img_size": 518,
    "patch_size": 14,
    "embed_dim": 1024,
    "enable_camera": False,
    "enable_camera_mlp": True,
    "enable_local_point": True,
    "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": {
            "bias": True,
            "head_dim": 1024,
            "inter_multi": 2,
            "base_lr": 0.01,
            "muon_update_steps": 5,
            "use_gate_fn": True,
        },
        "window_size": 1,
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--align-first-view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema", action="store_true")
    return parser.parse_args()


def collect_images(input_dir: Path, max_frames: int | None) -> list[Path]:
    images = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be positive")
        images = images[:max_frames]
    if not images:
        raise FileNotFoundError(f"No supported images in {input_dir}")
    return images


def load_model(checkpoint_path: Path, use_ema: bool, device: torch.device) -> ZipMap:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("ema") if use_ema and "ema" in checkpoint else checkpoint.get("model", checkpoint)
    model = ZipMap(**STREAMING_CONFIG)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval().to(device)


def align_to_first_view(extrinsics: torch.Tensor) -> torch.Tensor:
    """Express W2C poses in the coordinate system of frame zero."""
    count = extrinsics.shape[1]
    homogeneous = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device).repeat(1, count, 1, 1)
    homogeneous[:, :, :3, :] = extrinsics
    first_camera_to_world = torch.linalg.inv(homogeneous[:, :1])
    return (homogeneous @ first_camera_to_world)[:, :, :3, :]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ZipMap Streaming inference requires CUDA")
    if args.window_size < 1:
        raise ValueError("--window-size must be positive")
    image_paths = collect_images(args.input_dir, args.max_frames)
    device = torch.device("cuda")
    model = load_model(args.checkpoint, args.ema, device)
    images = load_and_preprocess_images([str(path) for path in image_paths]).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images, window_size=args.window_size)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    if args.align_first_view:
        extrinsics = align_to_first_view(extrinsics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        frame_names=np.asarray([path.name for path in image_paths]),
        extrinsics_world_to_camera=extrinsics[0].float().cpu().numpy(),
        intrinsics=intrinsics[0].float().cpu().numpy(),
        depth=predictions["depth"][0].float().cpu().numpy(),
        depth_conf=predictions["depth_conf"][0].float().cpu().numpy(),
    )
    summary = {
        "model": "ZipMap Streaming",
        "frame_count": len(image_paths),
        "image_shape": list(images.shape),
        "window_size": args.window_size,
        "align_first_view": args.align_first_view,
        "checkpoint": str(args.checkpoint),
        "output": str(args.output_dir / "predictions.npz"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
