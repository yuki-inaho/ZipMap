#!/usr/bin/env python3
"""Infer ZipMap and VGGT-Omega on the same frames, then create one Rerun RRD."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "_inference_comparison_worker.py"
VISUALIZER = ROOT / "scripts" / "visualize_reconstruction.py"


def extract_video(video: Path, output: Path, max_frames: int | None) -> Path:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    output.mkdir()
    index = 0
    try:
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(output / f"{index:06d}.png"), frame):
                raise RuntimeError(f"Could not write video frame {index}")
            index += 1
    finally:
        capture.release()
    if index == 0:
        raise RuntimeError(f"No frames found in video: {video}")
    return output


def worker_command(
    python: Path,
    model: str,
    input_dir: Path,
    checkpoint: Path,
    output: Path,
    max_frames: int | None,
    image_size: int | None,
    model_fp16: bool = False,
) -> list[str]:
    command = [
        str(python),
        str(WORKER),
        "--model",
        model,
        "--input-dir",
        str(input_dir),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
    ]
    if max_frames is not None:
        command += ["--max-frames", str(max_frames)]
    if image_size is not None:
        command += ["--image-size", str(image_size)]
    if model_fp16:
        command += ["--model-fp16"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path, help="Ordered RGB images")
    source.add_argument("--video", type=Path, help="Extract all video frames in order")
    parser.add_argument("--zipmap-checkpoint", type=Path, required=True)
    parser.add_argument("--omega-checkpoint", type=Path, required=True)
    parser.add_argument("--omega-repo", type=Path, required=True)
    parser.add_argument(
        "--omega-python", type=Path, help="Default: <omega-repo>/.venv/bin/python"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="One GPU ID runs sequentially; two IDs run concurrently",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--zipmap-image-size", type=int, default=518)
    parser.add_argument("--omega-image-size", type=int, default=384)
    parser.add_argument("--zipmap-model-fp16", action="store_true")
    args = parser.parse_args()

    if (
        len(args.gpus) not in (1, 2)
        or len(set(args.gpus)) != len(args.gpus)
        or min(args.gpus) < 0
    ):
        parser.error("--gpus must contain one or two distinct nonnegative IDs")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    omega_repo = args.omega_repo.resolve()
    # Preserve the venv's python symlink: resolving it would invoke /usr/bin/python
    # and silently drop all packages installed in the VGGT-Omega environment.
    omega_python = (args.omega_python or omega_repo / ".venv/bin/python").absolute()
    if not omega_python.is_file():
        parser.error(f"VGGT-Omega Python not found: {omega_python}")
    for checkpoint in (args.zipmap_checkpoint, args.omega_checkpoint):
        if not checkpoint.is_file():
            parser.error(f"Checkpoint not found: {checkpoint}")
    if args.input_dir and not args.input_dir.is_dir():
        parser.error(f"Input directory not found: {args.input_dir}")
    if args.video and not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    input_dir = (
        extract_video(args.video.resolve(), output / "frames", args.max_frames)
        if args.video
        else args.input_dir.resolve()
    )
    jobs = [
        (
            "zipmap",
            Path(sys.executable),
            ROOT,
            args.zipmap_checkpoint.resolve(),
            args.gpus[0],
        ),
        (
            "omega",
            omega_python,
            omega_repo,
            args.omega_checkpoint.resolve(),
            args.gpus[-1],
        ),
    ]
    processes = []
    for model, python, cwd, checkpoint, gpu in jobs:
        result_dir = output / model
        result_dir.mkdir()
        log_path = output / f"{model}.log"
        command = worker_command(
            python,
            model,
            input_dir,
            checkpoint,
            result_dir,
            None if args.video else args.max_frames,
            args.zipmap_image_size if model == "zipmap" else args.omega_image_size,
            args.zipmap_model_fp16 if model == "zipmap" else False,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if model == "zipmap":
            env.setdefault("TORCH_COMPILE_DISABLE", "1")
        print(f"Starting {model} on GPU {gpu}", flush=True)
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        processes.append((model, process, log_path))
        if len(args.gpus) == 1:
            code = process.wait()
            if code:
                raise SystemExit(f"{model} failed (exit {code}); see {log_path}")
    errors = []
    for model, process, log_path in processes:
        code = process.wait()
        if code:
            errors.append(f"{model}: exit {code} ({log_path})")
    if errors:
        raise SystemExit("Inference failed: " + "; ".join(errors))

    rrd = output / "comparison.rrd"
    command = [
        sys.executable,
        str(VISUALIZER),
        "--result",
        f"ZipMap={output / 'zipmap'}",
        "--result",
        f"VGGT_Omega={output / 'omega'}",
        "--output",
        str(rrd),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    summary = {
        "models": ["ZipMap Streaming", "VGGT-Omega-1B-512"],
        "gpu_ids": args.gpus,
        "input": str(args.video or args.input_dir),
        "frame_count": json.loads((output / "zipmap/runtime.json").read_text())[
            "frame_count"
        ],
        "rrd": str(rrd),
        "point_cloud_view": "fused all-frame reconstruction in each model tab",
        "pose_units": "arbitrary model scale",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
