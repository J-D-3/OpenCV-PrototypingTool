# Architecture

This document records the structure of the project and **where to make changes**
for common future work. It is kept short and updated as the design evolves.

## Layering: backend (`core/`) vs frontend (`ui/`)

The one hard rule:

> **`core/` must never import from `ui/` (or PyQt6). `ui/` depends on `core/`.**

This keeps the compute/graph logic headless and unit-testable, and confines all
Qt to the view layer. `operations.py` is verified Qt-free by the smoke test.

```
core/                 backend — pure Python, no Qt
  operations.py       the Operation registry (compute + metadata + param schema)
  graph.py            GraphModel: nodes + edges, the topology source of truth
  engine.py           DAG evaluator: topo order, dirty propagation, caching, error capture

ui/                   frontend — PyQt6 view layer
  controller.py       GraphController: bridges view items <-> model/engine
  image_utils.py      cv_to_qimage and (Phase 4) display normalization
  nodes.py            Node / ImageNode / FunctionNode / SaveToFileNode graphics items
  arrow.py            ArrowItem (edge graphics)
  canvas.py           GraphicsImageView + ImageDropWidget (the node canvas)
  viewer.py           ImageViewerWindow (per-node output inspector)
  parameters.py       (Phase 2) ParameterPanel auto-generated from a node's schema
  main_window.py      MainWindow (sidebar + canvas + parameter panel)

app.py                entrypoint (argparse + QApplication)
main.py               thin launcher -> app.main()
smoke_test.py         headless GUI safety net (QT_QPA_PLATFORM=offscreen)
engine_test.py        headless backend tests (no Qt at all)
```

## Data flow (how a change propagates)

1. A view item's `set_parameter` / `add_input_connection` calls the
   `GraphController` (it never touches other view items directly).
2. The controller mutates the `GraphModel` (edge added, param set) and marks the
   affected backend node — and everything downstream — dirty.
3. The `Engine` re-evaluates dirty nodes in topological order, caching clean
   ones, and records any failure on the node (`GraphNode.error`).
4. The controller refreshes exactly the recomputed view items
   (`refresh_from_model`) and fires `on_commit` side effects (e.g. save-to-file)
   only on committed — not preview — recomputes.

This replaced the old recursive scene-walking + re-entrancy flags.

### Internal dependency direction (no cycles)
`operations` ← `nodes` ← `arrow`, `viewer`, `canvas` ← `main_window` ← `app`.
`nodes` references `ArrowItem` only under `TYPE_CHECKING`, so `arrow`↔`nodes`
does not cycle at runtime.

## Where to change things

| I want to… | Go to |
|------------|-------|
| **Add an OpenCV operation** | Add one `Operation` entry in `core/operations.py` (id, label, category, ports, param schema, a `compute(inputs, params)` function). The sidebar and node factory pick it up automatically. |
| Give an op a non-image preview (e.g. draw contours) | Set its `render_preview(inputs, output, params) -> image` hook (consumed from Phase 4). |
| Show key stats for an op (e.g. #contours) | Set its `summary(output, params) -> dict` hook (consumed from Phase 4). |
| Change how a parameter control looks | `ui/parameters.py` (Phase 2); the widget is derived from the `ParamSpec.kind`. |
| Change node appearance / icons | `ui/nodes.py`. |
| Change canvas behaviour (drag, connect, grid) | `ui/canvas.py`. |
| Change the inspector window | `ui/viewer.py`. |
| Change evaluation / propagation | `core/engine.py`. |
| Change how the view drives the backend | `ui/controller.py`. |

## Run / test
```powershell
.\.venv\Scripts\Activate.ps1
python main.py [image]                 # launch the GUI
$env:QT_QPA_PLATFORM="offscreen"; python smoke_test.py   # headless checks
```
