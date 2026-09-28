#!/usr/bin/env python3
"""Visualize saved RGB/depth/poses in Rerun, including a camera-following 3D view.

Layout inspired by yuki-inaho/vggt-omega blackwell-develop's visualize.py.
Accepts predictions.npz + rgb.npy produced by the example review pipeline.
Coordinates: world-to-camera extrinsics, OpenCV camera axes (right, down, forward).
Depth and camera translations have arbitrary scale, not calibrated metres.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def load_result(folder: Path) -> dict:
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["rgb"] = np.load(folder / "rgb.npy", allow_pickle=False)
    for key in ("depth", "depth_conf"):
        if data[key].ndim == 4 and data[key].shape[-1] == 1:
            data[key] = data[key][..., 0]
    n, h, w = data["depth"].shape
    expected = {
        "rgb": (n, h, w, 3),
        "depth_conf": (n, h, w),
        "intrinsics": (n, 3, 3),
        "extrinsics_world_to_camera": (n, 3, 4),
    }
    for key, shape in expected.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(
                f"{folder}: invalid {key}: expected {shape}, got {data[key].shape}"
            )
    if not np.isfinite(data["depth"]).all() or (data["depth"] <= 0).any():
        raise ValueError("Depth must be finite and positive")
    homogeneous = np.tile(np.eye(4), (n, 1, 1))
    homogeneous[:, :3] = data["extrinsics_world_to_camera"]
    data["camera_to_world"] = np.linalg.inv(homogeneous)
    return data


def make_cloud(data: dict, percentile: float, stride: int, max_points: int):
    """Unproject depth using intrinsics of the actual preprocessed RGB image."""
    _, h, w = data["depth"].shape
    yy, xx = np.mgrid[:h:stride, :w:stride]
    points, colors = [], []
    for depth, conf, rgb, K, c2w in zip(
        data["depth"],
        data["depth_conf"],
        data["rgb"],
        data["intrinsics"],
        data["camera_to_world"],
    ):
        z = depth[::stride, ::stride]
        camera = np.stack(
            ((xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z), axis=-1
        )
        world = camera @ c2w[:3, :3].T + c2w[:3, 3]
        keep = conf[::stride, ::stride] >= np.percentile(conf, percentile)
        points.append(world[keep])
        colors.append(rgb[::stride, ::stride][keep])
    points, colors = np.concatenate(points), np.concatenate(colors)
    if len(points) > max_points:
        selection = np.linspace(0, len(points) - 1, max_points).astype(int)
        points, colors = points[selection], colors[selection]
    return points.astype(np.float32), colors


def model_layout(name: str):
    world = f"/{name}/world"
    cam = f"{world}/camera/image"
    return rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=cam, contents=[f"{cam}/rgb"], name="RGB"),
            rrb.Spatial2DView(
                origin=cam, contents=[f"{cam}/depth"], name="Depth (relative units)"
            ),
        ),
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=world,
                contents=[
                    f"{world}/cloud",
                    f"{world}/path",
                    f"{world}/frustums/**",
                    cam,
                ],
                name="World: point cloud + camera poses",
                line_grid=False,
            ),
            rrb.Spatial3DView(
                origin=world,
                contents=[f"{world}/cloud", cam],
                name="Point cloud from current camera",
                line_grid=False,
                eye_controls=rrb.EyeControls3D(tracking_entity=cam),
            ),
        ),
        name=name,
        row_shares=[0.45, 0.55],
    )


def log_result(name: str, data: dict, args) -> dict:
    world = f"{name}/world"
    n, h, w = data["depth"].shape
    rr.log(world, rr.ViewCoordinates.RDF, static=True)
    points, colors = make_cloud(
        data, args.conf_percentile, args.stride, args.max_points
    )
    rr.log(
        f"{world}/cloud",
        rr.Points3D(points, colors=colors, radii=rr.Radius.ui_points(1.5)),
        static=True,
    )
    poses = data["camera_to_world"]
    rr.log(
        f"{world}/path",
        rr.LineStrips3D([poses[:, :3, 3]], colors=[255, 130, 30]),
        static=True,
    )
    plane_distance = float(np.median(data["depth"])) * 0.04
    for i, (c2w, K) in enumerate(zip(poses, data["intrinsics"])):
        path = f"{world}/frustums/{i:03}"
        rr.log(
            path,
            rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
            static=True,
        )
        rr.log(
            path + "/image",
            rr.Pinhole(
                image_from_camera=K,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=plane_distance,
            ),
            static=True,
        )
    depth_range = np.percentile(data["depth"], [2, 98]).tolist()
    for i, (c2w, K, rgb, depth) in enumerate(
        zip(poses, data["intrinsics"], data["rgb"], data["depth"])
    ):
        rr.set_time("frame", sequence=i)
        camera = f"{world}/camera"
        rr.log(camera, rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        rr.log(
            camera + "/image",
            rr.Pinhole(
                image_from_camera=K,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=plane_distance,
            ),
        )
        rr.log(camera + "/image/rgb", rr.Image(rgb).compress(jpeg_quality=95))
        rr.log(
            camera + "/image/depth",
            rr.DepthImage(depth, depth_range=depth_range, colormap="Turbo"),
        )
    return {
        "frames": n,
        "rgb_shape": list(data["rgb"].shape),
        "points": len(points),
        "depth_units": "arbitrary, not calibrated metres",
        "cloud": "all frames fused; confidence-filtered; spatially subsampled",
        "camera_follow_entity": f"/{world}/camera/image",
        "w2c_c2w_inverse_max_error": float(
            np.abs(data["extrinsics_world_to_camera"] @ poses - np.eye(4)[:3]).max()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", action="append", required=True, metavar="NAME=DIRECTORY"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf-percentile", type=float, default=25)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument(
        "--spawn", action="store_true", help="Open the native Rerun viewer"
    )
    args = parser.parse_args()
    if not 0 <= args.conf_percentile <= 100 or args.stride < 1 or args.max_points < 1:
        parser.error("Invalid confidence percentile, stride, or max-points")
    results = {}
    for item in args.result:
        name, folder = item.split("=", 1)
        if not name.replace("_", "").replace("-", "").isalnum() or name in results:
            parser.error(
                "Names must be unique and contain letters, digits, underscores or hyphens"
            )
        results[name] = load_result(Path(folder))
    blueprint = rrb.Blueprint(
        rrb.Tabs(*[model_layout(name) for name in results], active_tab=0),
        rrb.TimePanel(timeline="frame", play_state="paused", fps=2, expanded=True),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
        auto_layout=False,
        auto_views=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("zipmap-vggt-omega-rgbd-review", spawn=False)
    rr.save(args.output, default_blueprint=blueprint)
    info = {name: log_result(name, data, args) for name, data in results.items()}
    rr.disconnect()
    if args.spawn:
        subprocess.Popen([sys.executable, "-m", "rerun", str(args.output.resolve())])
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
