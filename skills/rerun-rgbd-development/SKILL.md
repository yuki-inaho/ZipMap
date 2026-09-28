---
name: rerun-rgbd-development
description: Develop and verify Rerun viewers for RGB images, depth maps, point clouds, and camera poses, including synchronized timelines and camera-follow rendering. Use for RGB-D可視化, カメラ視点レンダリング, Rerun blueprints, and recording/debugging workflows.
---

# Rerun RGB-D development

Build a reusable viewer from an explicit geometry/data contract. Work with the
user's chosen models and datasets; this skill does not authorize new data sharing.

## Establish the contract

Inspect the existing producer and installed Rerun SDK before editing. Determine:

- RGB layout/range and the exact resize/crop/pad applied to it.
- Depth convention (camera Z or ray distance), units, validity mask, confidence.
- Intrinsics for the **processed** image, distortion status, and resolution.
- World-to-camera versus camera-to-world poses, axes, and frame ordering.

Read [geometry.md](references/geometry.md) when building or adapting the loader,
unprojection, or camera hierarchy. Do not silently rescale depth, assume metres,
or apply one model's preprocessing to another model's saved predictions.

## Implement the display

Choose the installed SDK's supported API, then lock the verified dependency in the
project's optional visualization environment. Keep model inference separate from
recording export, so the viewer can be debugged without rerunning a model.

A useful RGB-D layout contains RGB, depth, world geometry/cameras, and a 3D view
tracking the active pinhole camera. Log the active camera transform and images on
one sequence timeline. Record geometry's accumulation policy: an all-frame fused
cloud contains future frames and is not a causal map. Make confidence filtering,
point limits, and sampling explicit parameters rather than dataset constants.

A camera-follow view must render 3D points through the camera transform; a panel
that only displays the RGB image does not verify this behavior. Exclude RGB image
planes from that view if they hide the point cloud.

Read [viewer-verification.md](references/viewer-verification.md) for export,
launch, timeline inspection, and recovery from native/container rendering issues.

## Validate with reusable inputs

Use `scripts/make_rgbd_fixture.py --output <scratch-directory>` to produce an
analytic colored scene with known camera poses, RGB, camera-Z depth and K. It needs
only NumPy and emits `predictions.npz`, `rgb.npy`, and `fixture.json`; all data is
synthetic. The schema is described in [geometry.md](references/geometry.md).
The output directory must be empty; fixtures are never generated inside this skill.

Check a saved recording's readability, then **inspect rendered frames** at two or
more nonidentical camera poses. Verify image/depth/time synchronization, camera
tracking, and visibility of the point cloud. Keep numeric pose checks separate
from visual evidence. Synthetic success validates the adapter, not model accuracy.
Use authorized real data only when it is needed for the user's requested outcome.

## Keep private material out of staging

Before any `git add`, review the candidate file paths and contents. Stage only
explicitly named, reviewed source/instruction files. Never stage raw/private data,
model weights, credentials, environment dumps, recordings, screenshots, logs, or
machine/user-specific paths. `.gitignore` is not proof that a file is safe.

Do not copy examples from terminal history or local reports into the skill. Use
parameter names and synthetic examples. If a test exposes sensitive content, keep
it outside the tracked tree and report only what is needed. Publishing or uploading
a recording needs its own authorization; starting a local viewer does not grant it.

## Deliver

Provide the runnable command, recording path or live viewer URL, selected timeline,
and verification scope. Distinguish a completed recording from a live server and
leave only the user-facing service running. Include known rendering or geometry
limitations without claiming measurement accuracy from appearance alone.
