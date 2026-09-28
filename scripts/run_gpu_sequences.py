#!/usr/bin/env python3
"""Run independent image sequences on one or more GPUs in parallel."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "zipmap" if (ROOT / "zipmap").is_dir() else "omega"
RUNNER = ROOT / "scripts" / (
    "run_zipmap_streaming_sequence.py" if MODEL == "zipmap" else "run_omega_sequence.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Each immediate subdirectory is one independent sequence")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="Physical GPU IDs; default: 0. Use --gpus 0 1 for two GPUs")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("--gpus must contain unique, nonnegative IDs")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    sequences = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if not sequences:
        parser.error(f"no sequence directories in {args.input_root}")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    def run(gpu: int, sequence: Path) -> tuple[str, int]:
        output = args.output_root / sequence.name
        command = [sys.executable, str(RUNNER),
                   "--input-dir", str(sequence),
                   "--checkpoint", str(args.checkpoint.resolve()),
                   "--output-dir", str(output)]
        if args.max_frames is not None:
            command += ["--max-frames", str(args.max_frames)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if MODEL == "zipmap":
            env.setdefault("TORCH_COMPILE_DISABLE", "1")
        print(f"GPU {gpu}: {sequence.name}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output.mkdir(parents=True, exist_ok=True)
        (output / "run.log").write_text(result.stdout, encoding="utf-8")
        print(f"GPU {gpu}: {sequence.name}: exit {result.returncode}", flush=True)
        return sequence.name, result.returncode

    def worker(gpu: int, assigned: list[Path]) -> list[tuple[str, int]]:
        return [run(gpu, sequence) for sequence in assigned]

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, gpu, sequences[index::len(args.gpus)])
                   for index, gpu in enumerate(args.gpus)]
        results = [result for future in futures for result in future.result()]
    failures = [name for name, code in results if code]
    if failures:
        raise SystemExit(f"Failed sequences: {', '.join(failures)}. See each run.log")


if __name__ == "__main__":
    main()
