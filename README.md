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
run.bat [optional\path\to\image.png]      REM uses the venv automatically
REM or, with the venv active:
python main.py [optional\path\to\image.png]
```

Canvas controls: right-drag to connect nodes (drop on a full single-input node
to rewire it; **right-drag onto a node a source is already connected to to
disconnect it**), double-click to inspect a node, **mouse-wheel over a batch
node** to scroll its images, **Delete** to remove selected nodes/arrows, **S**
to swap a binary op's inputs, and **Save/Load Pipeline** (sidebar) to persist a
graph to JSON. Each node shows an operation-specific icon; batch nodes show an
`i/N` frame badge.

The **right-hand Inspector pane** always follows the selected node and stacks:
an image view (output / `render_preview`); a pixel-neighbourhood view (3/9/27/81
grid) that tracks the cursor over the image — showing each pixel's position and
per-channel value, left-click to freeze; and a histogram with per-channel
toggles and draggable min/max ranges that mask the image to in-range pixels.

---

## Project layout

The code is split into a **Qt-free backend (`core/`)** and a **PyQt6 frontend
(`ui/`)**; `core/` never imports `ui/`. See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full file map and "where to change things".

| Path | Role |
|------|------|
| `core/operations.py` | **Qt-free** operation registry. Each `Operation` is declared once (id, label, category, input/output ports, parameter schema, `compute(inputs, params)`, plus optional `render_preview`/`summary`/`out_space`/`space_aware`/`variadic`/`raw` hooks). The sidebar tree, node factory, parameter panel, evaluation, and inspection are all generated from this registry, so adding a function = adding one entry here. |
| `core/` | Backend: `graph.py` (topology), `engine.py` (DAG evaluator), `batch.py` (multi-image), `datatypes.py` (port types), `persistence.py` (JSON save/load). |
| `ui/` | Frontend: `controller.py`, `nodes.py`, `node_icons.py`, `canvas.py`, `inspector_pane.py`, `viewer.py`, `parameters.py`, `main_window.py`, `image_utils.py`, `arrow.py`. |
| `app.py` / `main.py` | Entrypoint (`app.py`) + thin launcher (`main.py`). |
| `smoke_test.py` / `engine_test.py` | Headless GUI safety net + Qt-free backend tests. |
| `output/` | Saved PNG outputs from the GUI (git-ignored) |
| `requirements.txt` | Runtime dependencies |

### Implemented pipeline
Image input → grayscale / BGR conversion → blur → (adaptive) threshold →
MSER region detection → arithmetic (sum / AND / diff of two images) → save to disk.
Changes propagate downstream automatically; a preview mode during slider drags
avoids spurious file writes.

---

## Project status (as of 2026-06-02)

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

- [x] **Phase 4 — typed data + generalized inspection.** Added
      `core/datatypes.py` (port types + permissive image compatibility, wired
      into connection validation); made `cv_to_qimage` robust (gray/BGR/BGRA +
      float normalization); made the inspector **signal-driven** (no polling)
      and able to show an op's `render_preview` image plus a `summary`
      key-facts line. A `run.bat` launcher was also added.
- [x] **Phase 5 — persistence + canvas editing.** Save/Load a whole pipeline
      to JSON (`core/persistence.py`; source images embedded as base64 PNG);
      delete nodes/edges (Delete), and swap a binary op's inputs (S).

- [x] **Phase 6 — op library + the three example workflows.**
  - color-quantization chain **To HSL → K-Means Cluster → Reduce Colors**
    (`CLUSTERS` payload; swatch preview + `clusters: k` summary);
  - segmentation chain **Resize → Blur → Adaptive Threshold → Find Contours →
    Filter Contours** (`CONTOURS` payload; Find/Filter draw the contours via
    `cv2.drawContours` for their preview and report the contour count);
  - Fourier chain **DFT → Inverse DFT** (`SPECTRUM` payload; DFT shows the
    log-magnitude spectrum; `idft(dft(img)) == img` is verified by a test).

- [x] **Extra ops + editing polish.** Added Gaussian Blur, Morphology, Canny,
      Sobel, Laplacian (Local Operations) and a Histogram node (new *Analysis*
      category, with a histogram-plot preview). Added **drag-to-rewire** (drop a
      new source on a full single-input node to re-point it) and **cycle
      prevention** on connections.
- [x] **Live inspector pane.** A docked, selection-following pane
      (`ui/inspector_pane.py`): image view + pixel-neighbourhood viewer
      (3/9/27/81, freeze on click) + per-channel histogram with range filtering
      that masks the preview.
- [x] **Batched multi-image processing.** "Open Images... (batch)" creates one
      source holding N images; the engine maps every op over the batch
      (`core/batch.py`), so one chain processes many images. Two-input ops
      zip/broadcast (e.g. diff every frame against one reference). The inspector
      header's `< i/N >` buttons (or the mouse wheel over a batch node) scrub
      which element all nodes preview; Save-to-File writes every element; batch
      sources are persisted.
- [x] **Create Batch node.** A variadic Input/Output node that assembles a
      batch from arbitrarily many image inputs (normalized to 3-channel BGR), so
      you can batch the outputs of separate sources/branches — without colliding
      with two-input ops like AND. (Enabled by `variadic`/`raw` op flags.)

- [x] **More ops + UX polish.** Geometry: **Resize** (scale + interpolation
      mode: AREA/LINEAR/CUBIC/…) and **Rotate** (angle + expand-canvas). Tone:
      **Normalize** (stretch / equalize / CLAHE; luminance-only for color),
      **Invert**, and **Local HDR** (Gaussian local mean/std normalization on
      luminance — radius/amplitude/strength). Clustering: **Auto Cluster**
      (auto-picks k from smoothed-histogram peaks, then k-means) and **Mean
      Shift** (`cv2.pyrMeanShiftFiltering`). Contours now use **stable id-based
      colors** (R G B C M Y) and **hierarchy/size ordering** (filled draws outer
      first), preserved across filtering. Save-to-File **falls back to a node's
      display preview** when it has no real image output (e.g. Filter Contours),
      via an `ANY` input port. Inspector refinements (name-based channel colors,
      wheel zoom-to-cursor, log histogram, range marker lines per channel,
      freeze/release on click); the pinned inspector window's parameter view was
      removed (params are edited only in the main window's panel). Open-Pipeline
      defaults to `./test/pipelines`.
- [x] **Disconnect by re-dragging.** Right-dragging a source onto a target it is
      already connected to **toggles the connection off** (removes the edge +
      arrow and re-evaluates downstream); the hover highlight shows an
      already-connected target as a valid drop. (`controller.is_connected`,
      `canvas._disconnect`.)

All planned phases are complete. Adding a new operation is cheap: define one
`Operation` in `core/operations.py` (+ optional `render_preview`/`summary`), and
the sidebar, parameter panel, evaluation, and inspection all follow — see
[ARCHITECTURE.md](ARCHITECTURE.md).

### Future ideas
- Undo/redo on the canvas; multi-input rewire; drag connection endpoints.
- More ops as needed (template matching, warps, feature detectors, …).
- Batch sources from a folder; per-element filenames carried through to Save.

### Known issues
- Reduce Colors / contour ops work in whatever space they receive; for the HSL
  chain, append **To BGR** to view the quantized result in true colors (the
  conversion is color-space aware, so it correctly maps HLS → BGR).

---

## Changelog
- **2026-06-03** — **Clustering diagnostics in the inspector preview.** The
  clustering nodes no longer show just a flat color swatch. **Auto Cluster** now
  draws *how k was chosen*: in peak mode, the original vs the smoothed /
  saturation-damped channel histogram the peaks were detected on, with each
  detected peak marked in the color it represents; in elbow mode, the inertia-vs-k
  curve with the chosen knee. **K-Means** shows a proportional palette (each
  cluster's width ∝ its pixel share), a feature-space scatter colored by cluster
  with ringed centers, and per-cluster spread (tightness) bars. All diagnostics are
  precomputed in `compute()` and stashed in the clusters payload, so the preview
  stays a cheap pure-draw pass. Also added **conditional parameters**
  (`ParamSpec.enabled_if`): a control grays out when it doesn't apply to the
  current mode — Auto Cluster's peak-detection params (channel / smoothing / min
  prominence) deactivate in elbow mode, so the panel shows at a glance which knobs
  are live. Suites: 33 smoke checks + 36 engine tests.
- **2026-06-02** — Added Geometry **Resize** (scale + interpolation mode) and
  **Rotate** (angle + expand); tone ops **Normalize** (stretch/equalize/CLAHE),
  **Invert**, and **Local HDR**; clustering ops **Auto Cluster** (histogram-peak
  k detection) and **Mean Shift**. Contours got stable id-based colors
  (R G B C M Y) and hierarchy/size ordering preserved across filtering;
  Save-to-File falls back to a node's display preview (`ANY` input port). Many
  inspector refinements (name-based channel colors, wheel zoom-to-cursor, log
  histogram, per-channel range marker lines, freeze/release); removed the pinned
  inspector window's parameter view; Open-Pipeline defaults to `./test/pipelines`.
  Added **disconnect-by-re-dragging** (right-drag onto an already-connected target
  toggles the edge off — `controller.is_connected` + `canvas._disconnect`).
  Suites: 23 smoke checks + 21 engine tests.
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
  each op's schema and removed the per-function control code. **Phase 4:**
  added `core/datatypes.py`, made `cv_to_qimage` robust (float normalization),
  and made the inspector signal-driven with `render_preview`/`summary` support.
  Added `run.bat`. **Phase 5:** JSON save/load of pipelines
  (`core/persistence.py`), node/edge deletion, and binary-op input swap.
  **Phase 6 (in progress):** added color-space conversions, K-Means Cluster
  (non-image clusters payload), and Reduce Colors — the first chain to flow
  non-image data and use the `render_preview`/`summary` hooks; then the
  segmentation chain Resize (Geometry) + Find/Filter Contours (Contours
  category), with contour previews drawn via `cv2.drawContours`; then the
  Fourier chain DFT / Inverse DFT (`SPECTRUM` payload, magnitude-spectrum
  preview, round-trip verified). Added Gaussian Blur, Morphology, Canny, Sobel,
  Laplacian, and a Histogram node (Analysis), plus drag-to-rewire and cycle
  prevention on connections. Added the live inspector pane. Collapsed the four
  conversion nodes into three space-aware ones (To Grayscale / To BGR / To HSL)
  that accept any input — the engine now tracks each node's color space.
