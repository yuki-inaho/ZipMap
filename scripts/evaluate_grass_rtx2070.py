#!/usr/bin/env python3
"""Evaluate ZipMap and VGGT-Omega predictions on held-out grass 3D anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_colmap_poses(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for line in (path / "images.txt").read_text().splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[0].isdigit():
            continue
        stem = Path(fields[9]).stem
        if not stem.isdigit():
            continue
        quat = [*map(float, fields[2:5]), float(fields[1])]
        poses[int(stem) - 1] = (
            Rotation.from_quat(quat).as_matrix(),
            np.array([float(value) for value in fields[5:8]]),
        )
    return poses


def pose_metrics(
    predicted: np.ndarray, reference: list[tuple[np.ndarray, np.ndarray]]
) -> dict:
    if len(predicted) < 2:
        return {"available": False, "reason": "fewer than two frames"}
    rotation_pred, translation_pred = predicted[:, :3, :3], predicted[:, :3, 3]
    rotation_ref = np.stack([item[0] for item in reference])
    translation_ref = np.stack([item[1] for item in reference])
    center_pred = -np.einsum("nji,nj->ni", rotation_pred, translation_pred)
    center_ref = -np.einsum("nji,nj->ni", rotation_ref, translation_ref)
    pred_steps = np.linalg.norm(np.diff(center_pred, axis=0), axis=1)
    ref_steps = np.linalg.norm(np.diff(center_ref, axis=0), axis=1)
    usable = pred_steps > 1e-8
    if not usable.any():
        return {"available": False, "reason": "zero predicted camera baseline"}
    scale = float(np.median(ref_steps[usable] / pred_steps[usable]))
    world_align = rotation_ref[0].T @ rotation_pred[0]
    aligned_centers = (
        center_pred - center_pred[0]
    ) @ world_align.T * scale + center_ref[0]
    position_error = np.linalg.norm(aligned_centers - center_ref, axis=1)
    c2w_pred = np.transpose(rotation_pred, (0, 2, 1))
    c2w_ref = np.transpose(rotation_ref, (0, 2, 1))
    aligned_orientations = np.einsum("ij,njk->nik", world_align, c2w_pred)
    delta = np.einsum("nji,njk->nik", aligned_orientations, c2w_ref)
    angle = np.degrees(Rotation.from_matrix(delta).magnitude())
    return {
        "available": True,
        "alignment": "first camera orientation/center and median consecutive baseline",
        "scale_to_colmap": scale,
        "median_position_error_sfm_units": float(np.median(position_error)),
        "median_orientation_error_degrees": float(np.median(angle)),
        "position_error_sfm_units": position_error.tolist(),
        "orientation_error_degrees": angle.tolist(),
    }


def map_anchors(
    xy: np.ndarray, original_size: np.ndarray, shape: tuple[int, int], model: str
) -> np.ndarray:
    width, height = map(int, original_size)
    out_height, out_width = shape
    if model == "zipmap":
        if height < width or out_height != out_width:
            raise ValueError("ZipMap evaluator currently expects padded portrait input")
        resized_width = round(width / height * out_height / 14) * 14
        pad_left = (out_width - resized_width) // 2
        x = (xy[:, 0] + 0.5) * resized_width / width - 0.5 + pad_left
        y = (xy[:, 1] + 0.5) * out_height / height - 0.5
    else:
        x = (xy[:, 0] + 0.5) * out_width / width - 0.5
        y = (xy[:, 1] + 0.5) * out_height / height - 0.5
    return np.stack([x, y], axis=1)


def evaluate_model(
    name: str,
    predictions_path: Path,
    anchors_dir: Path,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict:
    with np.load(predictions_path, allow_pickle=False) as data:
        names = data["frame_names"].tolist()
        extrinsics = data["extrinsics_world_to_camera"]
        depths = data["depth"].squeeze(-1)
    frames = [int(Path(filename).stem) - 1 for filename in names]
    if len(set(frames)) != len(frames) or any(frame not in poses for frame in frames):
        raise ValueError(f"{name}: duplicate or unregistered frame name")
    reference = [poses[frame] for frame in frames]
    rows = []
    for index, frame in enumerate(frames):
        with np.load(
            anchors_dir / f"{frame + 1:05d}.npz", allow_pickle=False
        ) as anchors:
            xy = anchors["xy_original"]
            z_sfm = anchors["z_sfm"]
            track_id = anchors["track_id"]
            original_size = anchors["original_size"]
        depth = depths[index]
        mapped = map_anchors(xy, original_size, depth.shape, name)
        sampled = cv2.remap(
            depth,
            mapped[:, 0].astype(np.float32).reshape(1, -1),
            mapped[:, 1].astype(np.float32).reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        ).ravel()
        valid = np.isfinite(sampled) & (sampled > 0) & (z_sfm > 0)
        train = valid & (track_id % 5 != 4)
        holdout = valid & (track_id % 5 == 4)
        if train.sum() < 20 or holdout.sum() < 5:
            raise ValueError(f"{name} frame {frame}: too few valid anchors")
        ratio = np.log(z_sfm[train] / sampled[train])
        center = np.median(ratio)
        mad = np.median(np.abs(ratio - center))
        inliers = np.abs(ratio - center) <= max(3 * 1.4826 * mad, 0.15)
        scale = float(np.exp(np.median(ratio[inliers])))
        error = np.abs(scale * sampled[holdout] - z_sfm[holdout]) / z_sfm[holdout]
        rows.append(
            {
                "frame": frame,
                "calibration_anchors": int(train.sum()),
                "holdout_anchors": int(holdout.sum()),
                "scale_to_colmap": scale,
                "holdout_median_relative_depth_error": float(np.median(error)),
                "holdout_p90_relative_depth_error": float(np.quantile(error, 0.9)),
                "frame_was_heldout_from_3d_fit": bool(frame % 5 == 4),
            }
        )
    return {
        "model": name,
        "frame_count": len(frames),
        "pose": pose_metrics(extrinsics, reference),
        "median_of_frame_depth_medians": float(
            np.median([row["holdout_median_relative_depth_error"] for row in rows])
        ),
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zipmap", type=Path, required=True, help="ZipMap predictions.npz"
    )
    parser.add_argument(
        "--omega", type=Path, required=True, help="Omega predictions.npz"
    )
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    poses = read_colmap_poses(args.colmap_model)
    report = {
        "coordinates": "arbitrary COLMAP SfM units",
        "depth_calibration": "robust per-frame scale from good CoW 3D tracks; track_id mod 5 == 4 held out",
        "models": [
            evaluate_model("zipmap", args.zipmap, args.anchors, poses),
            evaluate_model("omega", args.omega, args.anchors, poses),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
