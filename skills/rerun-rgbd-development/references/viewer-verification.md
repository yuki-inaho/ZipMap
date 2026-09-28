# Viewer verification and execution

## Select an API from evidence

Inspect `rerun.__version__`, relevant Python signatures, and `rerun --help` in the
actual visualization environment. Use official documentation matching that
version when needed. Camera controls, panels, sinks and CLI/MCP interfaces evolve.
A version used successfully elsewhere is a clue, not a requirement to upgrade.

Official references:

- https://rerun.io/docs/reference/types/archetypes/pinhole
- https://rerun.io/docs/concepts/logging-and-ingestion/transforms
- https://rerun.io/docs/reference/viewer/mcp

## Recording and service lifecycle

Save a `.rrd` with its default blueprint and flush/disconnect the recording before
loading it for verification. Do not assume `rr.save()` preserves a previously
connected live-viewer sink: verify the SDK's sink semantics. A simple reproducible
path is export first, then open the completed file with the matching viewer.

On SDKs that expose it, use `rerun rrd verify <recording>` for structural checks.
This does not prove the blueprint renders or follows the camera.

Native display and Web Viewer are alternatives. For a local Web Viewer, bind to
loopback and choose free ports. Read the printed connection URL: a bare web page
may load successfully without connecting to the recording service. Avoid changing
unrelated viewers or killing all processes named `rerun`.

Store process/session handles. When polling times out, recheck that same handle
or service before restarting. Stop an auxiliary verification viewer after its
checks; retain the user-facing viewer when the task calls for a live display.

## Inspect the actual display

At minimum inspect two frames with different poses:

- RGB and depth refer to the same frame and spatial footprint.
- The point cloud is actually visible; it is not replaced/covered by a photo.
- Camera-follow view changes pose with time; the world overview remains fixed.
- Tabs/views isolate separate model coordinate systems unless explicitly aligned.
- Relative units and filtering are stated, and omitted geometry is not disguised.

Record screenshots outside tracked source directories. A screenshot of the
entire desktop may expose unrelated private windows; prefer the app/view itself.

Recent versions provide `rerun viewer-mcp`. Discover its tool schemas at runtime;
use viewer state/time cursor/screenshot operations if present. Their payloads are
not stable enough to hardcode into the skill. A full SDK recording can be correct
while the UI backend fails, so inspect viewer logs as well as screenshots.

## Rendering failure fallback

If native window creation or screenshot capture reports invalid surface sizes,
GPU backend errors, or crashes inside a container, first inspect the actual error.
When supported, a headless viewer with an explicit modest viewport can verify the
same recording without the window manager; a local Web Viewer can provide the
interactive UI. Do not repeat a failed launch indefinitely or prescribe this
fallback for every desktop. Verify the fallback's rendered output and clearly
state which UI mode was tested.
