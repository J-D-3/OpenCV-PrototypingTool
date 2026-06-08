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
`core/optics_backend.py` lazily loads `optics_py` — a compiled pybind11 module from the
sibling **OPTICS-Clustering** repo (density clustering). It's optional and ABI-locked to
the building Python/platform; `load()` searches `$OPTICS_PY_DIR` then
`../OPTICS-Clustering/build-py/python/Release`, and raises a clear error (→ red node
border) if absent. The **Density Cluster** op (`core/operations.py`, id `hdbscan_cluster`)
has three `algorithm` modes — exact `hdbscan`, approximate `shdbscan` / `soptics` (CEOs
random projections, deterministic in `seed`, cosine/L2/L1 metric). It feeds a quantized
pixel cloud (the binding dedups internally, so `min_cluster_size` stays in *pixel* units)
and emits the standard `CLUSTERS` payload; noise (`-1`) maps to a trailing magenta centre.
The engine test skips cleanly when the binding is unavailable.

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
