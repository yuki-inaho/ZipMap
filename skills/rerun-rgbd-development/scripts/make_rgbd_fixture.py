#!/usr/bin/env python3
"""Generate only synthetic RGB-D: a colored box, floor, and moving pinhole camera.

Requires NumPy. No external files, network access, model, or GPU are used.
The output directory must be empty. Depth denotes camera Z, not ray distance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(vector):
    return vector / np.linalg.norm(vector)


def create_fixture(output: Path, width: int, height: int, frames: int):
    if width < 16 or height < 16 or frames < 2:
        raise ValueError("Use width/height >= 16 and at least two frames")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    K = np.array(
        [
            [0.9 * width, 0, (width - 1) / 2],
            [0, 0.9 * width, (height - 1) / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    v, u = np.mgrid[:height, :width]
    rays_camera = np.stack(
        ((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), axis=-1
    )
    lower = np.array([-0.5, -0.4, 2.6])
    upper = np.array([0.5, 0.6, 3.6])
    images, depths, extrinsics, positions = [], [], [], []
    max_pixel_error = 0.0
    max_surface_error = 0.0
    for cx in np.linspace(-0.6, 0.6, frames):
        center = np.array([cx, -0.25, 0.0])
        forward = normalize(np.array([0.0, 0.0, 3.0]) - center)
        right = normalize(np.cross([0.0, 1.0, 0.0], forward))
        down = np.cross(forward, right)
        rotation_cw = np.stack([right, down, forward], axis=1)
        directions = rays_camera @ rotation_cw.T
        # z=6 background, y=.7 floor, and axis-aligned box intersections.
        depth = (6 - center[2]) / directions[..., 2]
        labels = np.zeros((height, width), dtype=np.int8)
        floor = np.divide(
            0.7 - center[1],
            directions[..., 1],
            out=np.full_like(depth, np.inf),
            where=directions[..., 1] > 1e-12,
        )
        hit_floor = (floor > 0) & (floor < depth)
        depth = np.where(hit_floor, floor, depth)
        labels[hit_floor] = 1
        parallel = np.abs(directions) < 1e-12
        safe = np.where(parallel, 1.0, directions)
        t1 = (lower - center) / safe
        t2 = (upper - center) / safe
        near = np.where(parallel, -np.inf, np.minimum(t1, t2))
        far = np.where(parallel, np.inf, np.maximum(t1, t2))
        outside_parallel = (parallel & ((center < lower) | (center > upper))).any(
            axis=-1
        )
        enter = near.max(axis=-1)
        leave = far.min(axis=-1)
        hit_box = (~outside_parallel) & (enter > 0) & (leave >= enter) & (enter < depth)
        depth = np.where(hit_box, enter, depth)
        labels[hit_box] = 2
        world = center + depth[..., None] * directions
        checker = (np.floor(world[..., 0] * 5) + np.floor(world[..., 2] * 5)).astype(
            int
        ) % 2
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:] = [50, 90, 145]
        rgb[hit_floor] = np.where(
            checker[hit_floor, None] > 0, [190, 190, 190], [75, 75, 75]
        )
        rgb[hit_box] = [235, 175, 35]
        # Color actual box faces to reveal wrong rotation or axis conventions.
        box_face = np.argmin(
            np.minimum(np.abs(world - lower), np.abs(world - upper)), axis=-1
        )
        rgb[hit_box & (box_face == 0)] = [205, 60, 60]
        rgb[hit_box & (box_face == 1)] = [55, 185, 115]
        E = np.c_[rotation_cw.T, -rotation_cw.T @ center]
        # Known scene surfaces supplement the projection roundtrip.
        residual = np.where(
            labels == 0, np.abs(world[..., 2] - 6), np.abs(world[..., 1] - 0.7)
        )
        on_box = np.minimum(np.abs(world - lower), np.abs(world - upper)).min(axis=-1)
        residual = np.where(labels == 2, on_box, residual)
        camera = world @ E[:, :3].T + E[:, 3]
        projected = camera @ K.T
        projected = projected[..., :2] / projected[..., 2:]
        max_pixel_error = max(
            max_pixel_error, float(np.abs(projected - np.stack([u, v], axis=-1)).max())
        )
        max_surface_error = max(max_surface_error, float(residual.max()))
        images.append(rgb)
        depths.append(depth.astype("f4"))
        extrinsics.append(E.astype("f4"))
        positions.append(center.tolist())
    if max_pixel_error > 1e-7 or max_surface_error > 1e-7:
        raise AssertionError("Synthetic geometry consistency failure")
    np.save(output / "rgb.npy", np.stack(images))
    depth = np.stack(depths)
    np.savez_compressed(
        output / "predictions.npz",
        frame_names=np.array([f"synthetic_{i:04d}" for i in range(frames)]),
        depth=depth[..., None],
        depth_conf=np.ones_like(depth),
        intrinsics=np.repeat(K[None], frames, axis=0).astype("f4"),
        extrinsics_world_to_camera=np.stack(extrinsics),
    )
    metadata = {
        "synthetic": True,
        "scene": "axis-aligned box, checker floor, flat background",
        "depth_convention": "camera Z",
        "units": "synthetic scene units",
        "axes": "OpenCV right-down-forward",
        "width": width,
        "height": height,
        "frames": frames,
        "camera_centers": positions,
        "max_roundtrip_pixel_error": max_pixel_error,
        "max_analytic_surface_error": max_surface_error,
    }
    (output / "fixture.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    create_fixture(args.output, args.width, args.height, args.frames)


if __name__ == "__main__":
    main()
