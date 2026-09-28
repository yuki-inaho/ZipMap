---
name: gpu-inference-validation
description: Set up reproducible uv GPU inference and verify saved model outputs with example data, including single/multiple GPU execution, checkpoints, timing, memory, and reconstruction quality. Use for 推論環境整備, 実機検証, and checking whether inference results are meaningful, not for training optimization.
---

# GPU inference validation

Turn the user's requested runtime and output behavior into an executable,
inspectable result. Preserve their chosen models, checkpoint variants and scope.

## Revalidate the actual environment

Read repository instructions and current Git state. Inspect devices, free memory,
driver, interpreter, lockfile and installed framework build. GPU availability or
successful imports alone do not prove kernel support: run a small operation on
each selected device, then a real model forward. Verify accelerator compatibility
against installed architecture metadata and official sources when necessary.

Use the project's package manager and lock the tested configuration. Treat core
inference and optional visualization dependencies separately when practical.
Do not substitute CPU inference or an untrained model for a requested GPU/model
check without explicitly narrowing what the result proves.

## Execute the requested unit of work

Identify whether multi-GPU means independent sequences/jobs, tensor/model
parallelism, or distributed processing of one sequence. Resolve material ambiguity
before choosing an implementation. Assign a GPU to each independent worker and
preserve sequence order/state; arbitrary splitting can change temporal inference
and coordinate systems. Provide a valid single-device path when required.

Validate checkpoint/model compatibility rather than hiding missing weights with
`strict=False`. Record the variant and load errors. If access is denied, first
check whether the authorized credential is actually loaded, without printing its
value. A token's presence does not imply gated-repository approval. Do not request
or accept new licensing/access terms on behalf of the user.

Start with a manageable example, then finish the requested sequence scope. Keep
process handles and bounded logs so progress, OOM, failures and completion can be
verified. A timeout while observing a live process is not a reason to launch a
second copy. Distinguish inference time from load/save time and measure memory
with an explicitly named statistic.

## Validate usefulness, not only successful exit

Read [reconstruction-evidence.md](references/reconstruction-evidence.md) for
RGB-D/pose reconstruction. For other modalities, use corresponding output and
quality invariants. Check saved artifacts by reopening them, then assess image or
geometry consistency where relevant. Use synthetic fixtures for adapter tests and
authorized representative data for model behavior; these prove different things.

Do not silently change precision, resolution, frame selection or model variant to
make a check pass. If resource constraints require a change, preserve the failing
evidence and identify exactly what the successful run now establishes.

## Reusable, private-data-safe deliverables

Keep credentials, data, weights, recordings, logs, desktop captures, local account
names and private paths out of source, skill text and Git staging. Before staging,
review each candidate's content and use explicit file paths; never stage a whole
working directory with `git add .` or `git add -A`. Generated artifacts remain
local in ignored or external storage. Ignore rules are not a privacy classifier.

A reusable recipe takes paths/devices from arguments and demonstrates its contract
with public or synthetic examples. It does not embed a real user's dataset layout,
checkpoint location, service credentials or machine-specific defaults.

Report the run conditions, observed quality, limitations and artifact locations.
Only claim full completion when the user's requested execution scope and output
checks have actually completed. A smoke test should be identified as a smoke test.
