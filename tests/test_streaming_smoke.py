"""GPU smoke test for the Gradio-free ZipMap Streaming CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = REPOSITORY_ROOT / "examples" / "videos" / "drift-straight.mp4"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "checkpoint_online.pt"


def test_bundled_sequence_sample_is_available() -> None:
    assert SAMPLE_VIDEO.is_file()
    assert SAMPLE_VIDEO.stat().st_size > 0


def extract_two_frames(video_path: Path, output_dir: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened(), f"Could not open sample video: {video_path}"
        for index in range(2):
            ok, frame = capture.read()
            assert ok, f"Could not read sample frame {index}"
            assert cv2.imwrite(str(output_dir / f"{index:06d}.png"), frame)
    finally:
        capture.release()


@pytest.mark.gpu
def test_streaming_cli_smoke(tmp_path: Path) -> None:
    checkpoint = Path(os.environ.get("ZIPMAP_SMOKE_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.is_file():
        pytest.skip("Set ZIPMAP_SMOKE_CHECKPOINT or download checkpoints/checkpoint_online.pt")
    if not SAMPLE_VIDEO.is_file():
        pytest.skip(f"Missing bundled sample video: {SAMPLE_VIDEO}")

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    extract_two_frames(SAMPLE_VIDEO, image_dir)
    output_dir = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_zipmap_streaming_sequence.py",
            "--input-dir", str(image_dir),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--max-frames", "2",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "TORCH_COMPILE_DISABLE": "1"},
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    with np.load(output_dir / "predictions.npz") as predictions:
        assert predictions["frame_names"].shape == (2,)
        assert predictions["extrinsics_world_to_camera"].shape == (2, 3, 4)
        assert predictions["intrinsics"].shape == (2, 3, 3)
        assert predictions["depth"].shape[0] == 2
        assert np.isfinite(predictions["depth"]).all()
        assert np.isfinite(predictions["extrinsics_world_to_camera"]).all()
