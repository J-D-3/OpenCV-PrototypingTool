# OpenCV Prototyping Tool

A node-based visual programming GUI for prototyping OpenCV image-processing
pipelines. Drop an image onto the canvas, drop operation nodes from the sidebar,
wire them together with right-drag arrows, tune parameters with live sliders, and
view or save the result. Built with **PyQt6** + **OpenCV**.

---

## Getting started

### Prerequisites
- **Python 3.13** (the project's `__pycache__` and venv were built against 3.13;
  3.12 also works). On Windows the `py -3.13` launcher is the easiest way to pin it.
- Note: a `python` on `PATH` may resolve to a bundled interpreter (e.g.
  Inkscape's), which lacks these packages. Prefer the venv below.

### Setup (Windows / PowerShell)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run
```powershell
python main.py [optional\path\to\image.png]
```

---

## Project layout

| Path | Role |
|------|------|
| `operations.py` | **Qt-free** operation registry. Each `Operation` is declared once (id, label, category, input/output ports, parameter schema, `compute(inputs, params)`, plus optional `render_preview`/`summary` hooks for inspection). The sidebar tree and node factory are generated from this registry, so adding a function = adding one entry here. Importable and unit-testable without a GUI. |
| `node.py` | Qt node layer — `Node` base, `ImageNode`, a single Operation-driven `FunctionNode`, and a thin `SaveToFileNode` (side-effecting save) |
| `main.py` | GUI shell — main window, registry-generated sidebar tree, graphics canvas (grid snap, drag-drop, arrow creation), image viewer, parameter panels |
| `output/` | Saved PNG outputs from the GUI (git-ignored) |
| `requirements.txt` | Runtime dependencies |

### Implemented pipeline
Image input → grayscale / BGR conversion → blur → (adaptive) threshold →
MSER region detection → arithmetic (sum / AND / diff of two images) → save to disk.
Changes propagate downstream automatically; a preview mode during slider drags
avoids spurious file writes.

---

## Project status (as of 2026-06-01)

This is a **working prototype** being revived. The environment is now set up
(git + venv + pinned-ish deps), but the code carries prototype-era rough edges.

### Done
- [x] Git repository initialized.
- [x] Virtual environment (`.venv`, Python 3.13) created.
- [x] `requirements.txt` and `.gitignore` added.
- [x] Dependencies install cleanly; all modules import.
- [x] Headless smoke test (`smoke_test.py`) — builds the window offscreen and
      runs an image → grayscale → blur pipeline.
- [x] Resolved the `node.py` concatenation (removed the stray duplicate import
      block / self-import; it is now one clean module).
- [x] Removed dead/buggy code (`MainWindow.on_reset_zoom`, `ArrowItem.itemChange`).
- [x] Smoke test broadened into a refactor safety net (two-input nodes,
      parameter propagation, binary-op input order, save-to-file).
- [x] **Phase 1a — compute model separated from the view.** Introduced the
      Qt-free `operations.py` registry; collapsed the 10 `FunctionNode`
      subclasses into one Operation-driven node (+ thin `SaveToFileNode`);
      sidebar tree and node factory are now generated from the registry.
      Adding an OpenCV function is now ~one registry entry.
- [x] **Reorg** — split into `core/` (Qt-free backend) and `ui/` (PyQt6
      frontend) packages; see [ARCHITECTURE.md](ARCHITECTURE.md).
- [x] **Phase 1b — GraphModel.** Graph topology now lives in
      `core/graph.py` (nodes + edges); Qt items are thin views bound to it
      through `ui/controller.py`.
- [x] **Phase 3 — DAG evaluator.** `core/engine.py` does topological
      evaluation with dirty propagation, output caching, and per-node error
      capture (shown as a red border). Replaced the recursive scene-walking +
      re-entrancy flags. Covered by `engine_test.py` (Qt-free).
- [x] **Phase 2 — auto-generated parameters.** `ui/parameters.py` builds the
      controls from each op's `ParamSpec` schema (int/float sliders, enums,
      bools, text/path); deleted the ~370-line per-function `if/elif`. A new
      operation now needs zero UI code.

### Roadmap

Remaining phases toward the goal: rapidly wire OpenCV chains, expose every
parameter with live downstream updates, and inspect each node's result —
including ops whose output is not itself an image (e.g. findContours, drawn
back onto the input, with key stats like #contours shown in the GUI).

- **Phase 4 (next)** — typed data envelope (Image/Contours/Histogram/Labels/…) +
  type-dispatching, signal-driven inspector (image viewer, `render_preview`
  for non-image ops, `summary` key-info panel).
- **Phase 5** — save/load chains (JSON) + node/edge deletion & re-wiring.
- **Phase 6** — grow the library (Resize, GaussianBlur, FindContours,
  cvtColor→HSV/HSL, Histogram, KMeans, ColorQuantize, DFT/Fourier).

### Known issues
- Sidebar categories *Geometry* and *Fourier* are present-but-empty placeholders.
- No way to delete nodes or edges yet (the GraphModel supports it; no UI yet).
- `cv_to_qimage` does not normalize float / non-8-bit images (matters for Fourier).
- The inspector still polls (100 ms) for changes; it becomes signal-driven in Phase 4.

---

## Changelog
- **2026-06-01** — Revival started: git init, Python 3.13 venv, requirements,
  `.gitignore`, and this README added. Added headless smoke test. Refactored
  `node.py` (removed duplicate-import concatenation) and removed dead code
  (`on_reset_zoom`, `ArrowItem.itemChange`). Broadened the smoke test into a
  safety net. **Phase 1a:** added the Qt-free `operations.py` registry,
  collapsed the per-function node subclasses into one Operation-driven node,
  and generated the sidebar/factory from the registry. **Reorg:** split into
  `core/` (backend) and `ui/` (frontend) packages + ARCHITECTURE.md.
  **Phase 1b + 3:** moved graph topology into `core/graph.py` and added the
  `core/engine.py` DAG evaluator (topo order, dirty propagation, caching,
  error capture) wired through `ui/controller.py`; added `engine_test.py`.
  **Phase 2:** auto-generated the parameter panel (`ui/parameters.py`) from
  each op's schema and removed the per-function control code.
