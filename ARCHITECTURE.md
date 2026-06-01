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
  graph.py            (Phase 1b) GraphModel: nodes + edges, the topology source of truth
  engine.py           (Phase 3) DAG evaluator: topo order, dirty propagation, caching

ui/                   frontend — PyQt6 view layer
  image_utils.py      cv_to_qimage and (Phase 4) display normalization
  nodes.py            Node / ImageNode / FunctionNode / SaveToFileNode graphics items
  arrow.py            ArrowItem (edge graphics)
  canvas.py           GraphicsImageView + ImageDropWidget (the node canvas)
  viewer.py           ImageViewerWindow (per-node output inspector)
  parameters.py       (Phase 2) ParameterPanel auto-generated from a node's schema
  main_window.py      MainWindow (sidebar + canvas + parameter panel)

app.py                entrypoint (argparse + QApplication)
main.py               thin launcher -> app.main()
smoke_test.py         headless safety net (run with QT_QPA_PLATFORM=offscreen)
```

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
| Change evaluation / propagation | `core/engine.py` (Phase 3). |

## Run / test
```powershell
.\.venv\Scripts\Activate.ps1
python main.py [image]                 # launch the GUI
$env:QT_QPA_PLATFORM="offscreen"; python smoke_test.py   # headless checks
```
