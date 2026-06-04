# CLAUDE.md — working notes for this project

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

## Run / test (Windows PowerShell)
```powershell
.\.venv\Scripts\python.exe main.py [image]                       # launch GUI
$env:QT_QPA_PLATFORM='offscreen'; .\.venv\Scripts\python.exe smoke_test.py   # GUI safety net
.\.venv\Scripts\python.exe engine_test.py                        # Qt-free backend tests
```
Both suites must stay green before committing. Counts as of 2026-06-04:
**34 smoke checks + 40 engine tests.**

## Commit convention
- Branch first if on the default branch; commit/push only when asked.
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

## Canvas gestures
Right-drag = connect; drop on a full single-input node = rewire; **right-drag
onto a node a source is already connected to = disconnect (toggle)**. Double-click
inspects; wheel over a batch node scrubs frames; Delete removes; S swaps a binary
op's two inputs.
