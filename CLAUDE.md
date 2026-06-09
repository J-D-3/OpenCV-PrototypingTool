# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

OpenCVPrototypingTool: a PyQt6 node-graph GUI for prototyping OpenCV
image-processing chains. See `README.md` (status/changelog) and
`ARCHITECTURE.md` (structure + where to change things) first.

## The one hard rule
`core/` is **Qt-free** and never imports from `ui/` (or PyQt6). `ui/` depends on
`core/`, never the reverse. The smoke test verifies `core/` imports headless.

## Adding an OpenCV operation = one registry entry
Declare one `Operation` in `core/operations.py` (id, label, category, input/output
`Port`s, `ParamSpec` params, `compute(inputs, params)`, optional
`render_preview` / `summary` / `out_space` / `space_aware` / `variadic` / `raw`).
The sidebar tree, node factory, parameter panel, evaluation, and inspection all
follow automatically. Add a corner glyph by keying the op id in
`ui/node_icons.py` (`ICON_BY_OP` -> a `_DRAWERS` key).

## Setup (first time / fresh clone)
The `.venv` is git-ignored, so a fresh clone has none — create it before running
anything. Use Python 3.13 (3.12 and 3.14 also work); a bare `python` on PATH may
resolve to a bundled interpreter (e.g. Inkscape's) that lacks the packages — prefer
the venv.
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run / test (Windows PowerShell)
```powershell
.\.venv\Scripts\python.exe main.py [image]                       # launch GUI (or run.bat [image])
$env:QT_QPA_PLATFORM='offscreen'; .\.venv\Scripts\python.exe smoke_test.py   # GUI safety net
.\.venv\Scripts\python.exe engine_test.py                        # Qt-free backend tests
```
Both suites must stay green before committing. Each is a single script that runs
all its checks (no test framework / per-test selection — narrow by commenting out
calls if needed). Counts as of 2026-06-08: **36 smoke checks + 47 engine tests.**

## Optional OPTICS backend (Density Cluster node)
`core/optics_backend.py` lazily loads the **`optics`** package — a pip-installable pybind11
core (`_optics`) + high-level colour API from the sibling **OPTICS-Clustering** repo. Install
with `pip install <repo>/python` (needs a C++20 compiler + CMake); it's ABI-locked to the
building Python/platform. `load()` imports `optics`, falling back to `$OPTICS_PY_DIR` / the
sibling `OPTICS-Clustering/python` dir, and raises a clear error (→ red node border) if absent.
The **Density Cluster** op (`core/operations.py`, id `hdbscan_cluster`) is driven by
`optics.cluster_image`, which **dedups, voxel-quantizes, and converts sRGB→CIELAB internally**
— the node just hands it the BGR image. Five `algorithm` modes map to cluster_image's `algo`
(exact `hdbscan` / `optics-xi` / `optics-threshold`, approximate `shdbscan` / `soptics`). Noise
(`-1`) is reassigned to the nearest cluster (`noise_handling="nearest"`, default — a usable
quantization) or flagged magenta (`"flag"`). Note this CIELAB is *true* CIE (L 0–100), not
cv2's 8-bit Lab, so voxel/size values differ from the old `optics_py` binding (≈2 Lab / 4 RGB
voxel; ≥8 over-merges). Density clustering needs *separated* colour modes; on smooth photos it
labels most pixels noise, so K-Means / Auto Cluster are better quantizers there. The engine
test skips cleanly when the package is unavailable.

## Commit convention
- **Commit locally and automatically** on every atomic change (precise, concise
  message), keeping both test suites green across commits. **Push only after the
  user confirms** — suggest it after a coherent arc of commits.
- When a commit message contains double quotes, PowerShell's native-arg parsing
  breaks `git commit -m`. Use a here-string (`git commit -m @'...'@`) or write
  `_commitmsg.txt` and `git commit -F _commitmsg.txt` (the file is git-ignored).
- The exit-255 you sometimes see on commit is just Git's LF->CRLF warnings on
  stderr — check `git log` to confirm the commit actually landed.
- Stage only the specific files you changed (never `git add -A` — scratch files
  and user pipelines live in the tree). End messages with the Co-Authored-By line.

## Key concepts (quick map)
- `core/graph.py` GraphModel = topology source of truth (nodes + edges, dirty
  marking, `topo_order`, `creates_cycle`, `inputs_of`, `incoming`).
- `core/engine.py` Engine = topological eval, caching, dirty propagation, per-node
  error, color-space tracking, Batch auto-fan-out.
- `core/batch.py` Batch = a stack of images on one edge; engine maps ops over it,
  zip/broadcast for multi-input.
- `ui/controller.py` GraphController = the only place the frontend drives the
  backend. `can_connect` / `can_rewire` / `is_connected` / `delete_edge` /
  `replace_input` / `set_param` / `set_preview_index`.
- Color-space tracking: `GraphNode.color_space` via `op.out_space`
  (`bgr`/`gray`/`hls`/`binary` fixed, `passthrough`, `auto`); `space_aware` ops
  (the 3 conversions) get the input space as a 3rd `compute` arg.
- `core/codegen.py` = walks the graph upstream from a node into language-neutral
  pseudocode (`cv::`-style). Feeds the **Export Code** node (full pipeline) and the
  Function-info tooltip (single op). A node that's a single OpenCV call should add a
  `_CODE` emitter so the generated line is precise; tested by `test_codegen*`.

## Canvas gestures
Right-drag = connect; drop on a full single-input node = rewire; **right-drag
onto a node a source is already connected to = disconnect (toggle)**. Double-click
inspects; wheel over a batch node scrubs frames; Delete removes; S swaps a binary
op's two inputs. Selecting a node turns it **yellow** and its whole data flow
(every ancestor + descendant, nodes and edges) **green** — see
`canvas._update_flow_highlight`.
