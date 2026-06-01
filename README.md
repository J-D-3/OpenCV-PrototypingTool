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
| `main.py` | GUI shell — main window, sidebar function tree, graphics canvas (grid snap, drag-drop, arrow creation), image viewer window, per-function parameter panels |
| `node.py` | Node model — `Node` base, `ImageNode`, `FunctionNode`, and the operation nodes (SaveToFile, Blur, Threshold, AdaptiveThreshold, ToGrayscale, ToBGR, Sum, AND, Diff, MSER) |
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

### Known issues / cleanup backlog
- **Unfinished UI:** sidebar categories *Geometry* and *Fourier* are listed but
  have no functions. There is no way to **delete** nodes or arrows.
- The smoke test covers happy-path wiring only; no coverage of two-input nodes
  (Sum/AND/Diff), parameter changes, or save-to-file.

### Next steps (proposed)
1. Either implement or remove the empty Geometry/Fourier categories.
2. Add node/arrow deletion to the canvas.
3. Broaden the smoke test (two-input nodes, parameters, save-to-file).

---

## Changelog
- **2026-06-01** — Revival started: git init, Python 3.13 venv, requirements,
  `.gitignore`, and this README added. Added headless smoke test. Refactored
  `node.py` (removed duplicate-import concatenation) and removed dead code
  (`on_reset_zoom`, `ArrowItem.itemChange`).
