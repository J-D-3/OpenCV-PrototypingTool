# OpenCV Prototyping Tool

A node-based visual programming GUI for prototyping OpenCV image-processing
pipelines. Drop an image onto the canvas, drop operation nodes from the sidebar,
wire them together with right-drag arrows, tune parameters with live sliders, and
view or save the result. Built with **PyQt6** + **OpenCV**.

---

## Getting started

### Prerequisites
- **Python 3.13** (the project's `__pycache__` and venv were built against 3.13;
  3.12 and 3.14 also work — 3.14 resolves wheels for all deps and both test suites
  pass). On Windows the `py -3.13` launcher is the easiest way to pin it.
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
- **2026-06-11** — **Density Centers: density clustering as a centre detector.** New
  **Density Centers** node (`image → CENTERS`) runs the same OPTICS/HDBSCAN clustering as
  **Density Cluster** but emits only the discovered colour modes as CIELAB seeds (straight
  from the library's `palette[*].lab`), so it feeds **Assign to Centers** exactly like
  **Detect Color Centers** — density picks *which* colours exist, Assign labels every pixel
  in unified Lab (with the luminance-weight / k-means knobs), and the sparse "bridge" colours
  density calls noise fall to their nearest centre at assignment time. This replaces the
  Density Cluster node's own BGR nearest-noise pass for that workflow. **Note:** it's
  density-detected centres + Voronoi assignment, which reassigns *all* pixels (not just
  noise), so it differs from Density Cluster's density *segmentation* — kept as a separate
  node (rather than changing Density Cluster's output type) so existing pipelines and the
  reachability diagnostic are untouched. Same density knobs (algorithm, sizes, colour space,
  voxel, approx metric/seed); no noise-handling param. The inspector shows the same
  interactive 3-D Lab scatter as Detect Color Centers (payload-driven). Suites: 36 smoke +
  52 engine.
- **2026-06-10** — **Detect Color Centers: merge split centres (chroma-cut fix) + isolated-peak
  basin fix.** A bright/pale cluster straddling the **Neutral chroma C\*** cut splits into a hue
  centre *and* a lightness centre at nearly the same Lab point — which Assign then renders as two
  clusters (k-means *refine* pushing the near-identical seeds apart). New **Merge distance ΔE**
  param re-fuses any two detected centres within that ΔE: one colour mode, however it was found.
  The merged centre is the **support-weighted mean** of its parts (exactly the centroid of their
  pixel union) and inherits *both* histogram components. The lower-support part folds into the
  higher; in the preview the surviving centre is drawn **solid** and the merged-away one a
  **dashed** line with a hollow marker (at its own original peak + colour), so you can still see
  what was merged and where. Default 12 ΔE; 0 disables. Also fixed a latent bug in the
  scatter's peak-basin walk: an **isolated** hue peak on a near-zero plain returned the *complement*
  arc (excluding the peak), so half a cluster showed as scatter noise — the descent now stops at a
  near-zero floor, not only when the histogram rises. Suites: 36 smoke + 49 engine.
- **2026-06-09** — **Density Cluster: reachability in true colours + cluster axis.** The
  reachability bars are now painted each ordered point's **own original colour** (so the
  within-cluster colour variation is visible — e.g. several distinct reds that all belong
  to one cluster), with a **cluster axis** beneath them: each ordered point in its cluster's
  **mean** colour (matching the "extracted colours" swatch) and **tick marks at every
  cluster-run boundary**. Together they explain why many distinct-looking colour runs map to
  few clusters — the bars vary, the axis stays one colour per cluster.
- **2026-06-09** — **Auto Cluster split into Detect Color Centers → Assign to Centers
  (CIELAB/LCh).** Auto Cluster's 12 parameters and tangled HLS+chroma+Lab logic were
  replaced by two composable nodes that work in one perceptual space — true CIELAB / LCh
  (via the existing pure-NumPy `_srgb_to_lab`, the same space as Density Cluster and the
  inspector scatter). **Detect Color Centers** (`image → CENTERS`) finds cluster *seeds* —
  count **and** colour: chromatic pixels by a chroma-weighted circular **hue** histogram,
  neutral pixels (C\* below a threshold) by a **lightness L\*** histogram with an *adaptive*
  number of gray levels (no more fixed `gray_levels`). A **Min cluster area** knob discards
  centres backed by less than a set fraction of the image — a size floor that complements
  prominence (which only asks "is this a distinct mode?"), filtering tiny-but-locally-prominent
  specks without a unit-dependent height threshold. The preview's histogram markers track the
  *surviving* centres, so a peak dropped by Min cluster area (or Max centres) loses its indicator
  too — just like one dropped by Min peak prominence. **Assign to Centers**
  (`image + CENTERS → CLUSTERS`) labels every pixel against those seeds in unified 3-D
  CIELAB — **nearest centre (ΔE)** or **k-means refined *from* the detected centres**
  (`KMEANS_USE_INITIAL_LABELS`, so the detection guides the result instead of a random
  init). Because assignment is unified, the detection-time chroma split only shapes *where
  seeds come from*, never *where a pixel lands* — a pastel straddling the cut goes to
  whichever centre is truly closest, so the old "sharp cylinder slices a real cluster"
  problem dissolves. Each detected centre's colour is now *used*, not thrown away. The old
  `auto_cluster` op (and its elbow mode) was removed — re-wire saved pipelines to the new
  pair. Suites: 36 smoke + 46 engine.
- **2026-06-09** — **Cluster scatter: CIELAB axes; collapsible inspector layout.** The 3-D
  colour scatter now draws the full CIELAB solid: the **L axis** as a black→white gradient up
  the middle, an **a axis** (green→red) and **b axis** (blue→yellow) crossing at neutral
  (gradient lines, labelled), and the background gradient **inverted** (light top, dark bottom)
  to match the L axis. Layout: the three panes (menu / pipeline / inspector) are in a splitter
  with **Show: Menu | Pipeline | Inspector** toggle buttons — collapse any pane and the rest
  take its width; the **pipeline pane's minimum width dropped to ~2/3** (the hint label now
  word-wraps), and the inspector can grow wider for the scatter.
- **2026-06-09** — **Density Cluster: richer preview + interactive 3-D colour scatter.** The
  node preview stacks (1) the recoloured image titled with the **cluster count**, (2) an
  **extracted-colours palette** (swatch width ∝ pixel share, like Auto Cluster), and — when
  enabled — the **reachability plot with a cluster ribbon** beneath it (each ordered point's
  cluster colour, aligned with the bars). The **3-D colour-space scatter** moved into the
  **inspector pane** as an *interactive* widget (drag to rotate, scroll to zoom): pixels
  plotted in perceptual **CIELAB** space, framed by a faint reference **colour sphere** with a
  **hue-rainbow equator** (where each hue sits in the a–b plane) and the neutral **L (gray)
  axis**, over a **gray gradient backdrop** (dark top / light bottom) so near-black *and*
  near-white clusters stay visible. Points are painted their cluster mean, with toggles for
  **per-cluster enclosing spheres** and **true colour** (each pixel its own colour). It's shown
  for **all the clustering nodes** — Density Cluster, K-Means, and Auto
  Cluster — which now hide the pixel-neighbourhood grid + histogram (meaningless for a palette)
  and show the interactive scatter in their place (the sRGB→CIELAB conversion is a local
  pure-NumPy helper, so K-Means / Auto Cluster need no extra dependency). Also fixed the
  reachability plot using `min_cluster_size` as its `min_pts`
  (which over-smoothed it into a featureless ramp) — it now uses a small `min_pts` so density
  peaks are visible.
- **2026-06-09** — **Density Cluster: migrated to the new `optics` colour API.** The sibling
  OPTICS-Clustering library replaced its raw `optics_py` binding with a pip-installable
  **`optics`** package (`pip install <repo>/python`) whose high-level `optics.cluster_image`
  **dedups, voxel-quantizes, and converts sRGB→CIELAB internally**. The node now drives that
  one call instead of doing its own cv2-Lab conversion + manual voxel/dedup. Net effects: the
  colour conversion is now *true* CIELAB (L 0–100, perceptually correct — the old cv2 8-bit Lab
  was an approximation); algorithm modes map 1:1 to cluster_image (`hdbscan`, `optics-xi`,
  `optics-threshold`, `shdbscan`, `soptics`); and the fine knobs cluster_image fixes as good
  defaults (`min_samples`/`method`/`threshold`/`chi`) were dropped, leaving **Min cluster size**
  + **Min region size** (`min_cluster_frac`) as the main controls. Colour space is now Lab/RGB
  (HLS/LCh removed — the study found Lab best). Because the Lab/voxel scales changed, saved
  Density-Cluster pipelines need their `voxel_bin` (now ~2 Lab / 4 RGB; ≥8 over-merges) and
  size values re-tuned. `core/optics_backend.py` now imports the `optics` package. Suites:
  36 smoke + 47 engine.
- **2026-06-08** — **Density Cluster: OPTICS-exact mode, noise handling, searchability.**
  Three fixes prompted by a real-photo run. (1) Added the **OPTICS (exact)** algorithm
  (the binding already exposed `cluster_threshold`/`extract_xi`; it just wasn't wired as a
  mode) — so all four are selectable: OPTICS, HDBSCAN, sOPTICS, sHDBSCAN. (2) A **Noise
  pixels** param: *Assign to nearest cluster* (default) folds sparse pixels into the closest
  colour for a usable K-Means-like quantization, or *Flag colour* paints them magenta to
  reveal what was sparse. This fixes the "whole image turns pink" surprise on smooth photos,
  where density clustering correctly finds few separated modes and labels most pixels noise
  (for such images K-Means / Auto Cluster remain the better quantizers). (3) The node is now
  found by searching **"OPTICS" / "HDBSCAN" / "sOPTICS"** (op descriptions joined the search
  haystack). Suites: 36 smoke + 47 engine.
- **2026-06-08** — **Density Cluster: reachability-plot preview.** A "Show reachability
  plot" toggle computes the OPTICS **reachability** of the colour cloud and stacks the
  classic plot under the node preview — ordered points as bars (height = reachability),
  coloured by cluster, so **valleys are clusters** and tall peaks are the gaps/noise
  between them. It's the density landscape behind the clustering, useful for tuning the
  minimum size / threshold by eye. Precomputed in `compute()` (cheap pure-draw preview);
  off by default (costs one extra OPTICS pass on the unique colours).
- **2026-06-08** — **Density Cluster: sHDBSCAN + sOPTICS modes.** The density-clustering
  node (formerly "HDBSCAN Cluster"; op id unchanged) gained an **Algorithm** selector:
  exact **HDBSCAN\***, or the scalable approximate **sHDBSCAN** / **sOPTICS** (CEOs
  random projections — faster on big colour clouds, deterministic in a **seed**, with a
  cosine/L2/L1 **metric**; cosine clusters by colour *direction*, so different brightnesses
  of one hue merge). sOPTICS adds a threshold-vs-**Xi** extraction choice (defaults to Xi —
  the flat cut over-segments without hand-tuning). Mode-specific params gray out per
  algorithm. Required extending the sibling `optics_py` binding to expose `shdbscan` /
  `soptics`. Suites: 36 smoke + 47 engine.
- **2026-06-08** — **HDBSCAN Cluster node (density colour clustering).** A new
  *Color Quantization* op backed by the sibling **OPTICS-Clustering** library's
  `optics_py` binding (which we extended to expose HDBSCAN\*). Unlike K-Means it needs
  **no k** — just `min_cluster_size` — finds colour modes at differing densities, and
  labels sparse anti-aliasing/JPEG "bridge" colours as **noise** (painted a magenta
  flag). Params: min cluster size, min samples, EOM/Leaf selection, colour space
  (Lab / BGR / HLS / LCh), a voxel-quantize speed knob, and a min-cluster fraction.
  It emits the standard `CLUSTERS` payload, so **Reduce Colors** and the cluster
  diagnostics consume it like K-Means. The binding is **optional**: `core/optics_backend.py`
  loads it defensively (searches `$OPTICS_PY_DIR` then the sibling repo's build dir) and
  surfaces a clear error if it's missing — the node degrades, the app never crashes.
  Suites: 36 smoke + 47 engine.
- **2026-06-08** — **View-layer errors now surface in the UI.** Failures in the
  view layer — `render_preview` / `summary` hooks, save-to-file, and export-code —
  previously only printed to the console, so a failed save or preview looked like
  success. They now travel through a new `ControllerSignals.notify(level, message)`
  channel to a **status bar** (errors red and lingering; successful saves/exports
  shown briefly as info), matching how backend compute errors already show a red
  node border. Also fixed a **mojibake** bug (the Image inspector's size line showed
  garbage instead of `W×H`) and removed some dead code in `ui/canvas.py`. Suites:
  36 smoke + 46 engine.
- **2026-06-04** — **Auto Cluster: chroma-based achromatic handling.** Hue-peak
  detection is now weighted by **chroma** (max−min of BGR), not HLS saturation —
  because HLS S wrongly reads near-white/near-black pixels as fully saturated (the
  double-cone narrows), whereas chroma is ~0 for white, gray AND black. New
  **Separate gray/white/black** option pulls low-chroma pixels out and clusters them
  by lightness (into `gray_levels` clusters), so desaturated regions stop falling
  into the coloured clusters; while enabled, the hue histogram is hard-gated to
  chromatic pixels so achromatic regions can't form phantom hue clusters. The
  inspector histogram also now labels L/S as **%**. Suites: 35 smoke + 46 engine.
- **2026-06-04** — **Resize: "longer edge → length" mode.** Resize gained a `mode`:
  the existing scale factor, or **fixed** — scale so the image's longer edge becomes
  a target `length` px (aspect preserved), to normalize varying input sizes. Works on
  contours too (scaled by `length / longer edge of the reference shape`). The
  irrelevant field (Scale / Length) grays out per mode.
- **2026-06-04** — **Auto Cluster (elbow): a "k nudge" knob.** Elbow mode gained a
  `k_bias` parameter that offsets the auto-detected k by N clusters relative to the
  inertia knee (+ for more, − for fewer; 0 = the plain knee), clamped to
  [2, Max clusters]. A direct index offset (not a score tilt — the inertia curve is
  flat past the knee, so a tilt would jump straight to max). Elbow-only (grayed out
  in peaks mode).
- **2026-06-04** — **Deskew & Crop: negative border trims inward.** The Border
  parameter now accepts negative values (range [-100, 100], default 0): a negative
  border crops *tighter* than the contour's box (trims the edges) instead of only
  padding outward.
- **2026-06-04** — **Auto Cluster peak detection: mean-valley prominence + flat-step
  reject.** A histogram mode now counts when it rises above the **mean of its two
  surrounding valleys** by `min_prominence` of its height (was: above the *higher*
  valley) — so a sub-peak nested in a "mountain range" (e.g. the 5 in
  `0,0,3,5,4,7,8,3,0`) is detected, not dropped for the shallow side facing the
  taller peak. A peak must also **dip on both sides**, so the shoulder of a
  quasi-flat step isn't counted. `min_prominence` default 0.2 → 0.3 (the mean
  measure is more sensitive). Suites: 35 smoke + 43 engine.
- **2026-06-04** — **Resize scales contours; Largest Contour draws bold outlines.**
  **Resize** is now polymorphic (ANY in/out): fed a Contours payload it scales the
  contour coordinates (and the reference shape + preview background) instead of an
  image — so you can segment on a downscaled image and map the contours back onto
  the full-resolution original (e.g. Resize ↓ → segment → Find Contours → Resize ↑
  → Deskew & Crop the original). `datatypes.ANY` is now a wildcard both ways.
  **Largest Contour** draws the kept contours' outlines boldly on a dimmed backdrop
  so the selection is obvious (was a near-invisible 1px line on a white blob).
  Suites: 35 smoke + 42 engine.
- **2026-06-04** — **Two histogram-driven ops: Auto Threshold + Backproject.**
  **Auto Threshold** (Thresholding) picks the cut level automatically — Otsu,
  Triangle, or Valley (deepest dip between the two largest histogram modes) — and
  outputs a binary mask (with optional invert). **Backproject** (Analysis) is a
  two-input op (target image + a Histogram-node model) that projects the model's
  distribution onto the target to produce a likelihood map (bright where the image
  matches the modelled colour); `chroma_only` matches on Hue+Saturation for
  lighting robustness. The Histogram node's Hue smoothing is now circular (0/179
  join), so reds smooth correctly. Suites: 34 smoke + 40 engine.
- **2026-06-04** — **Fullscreen start, sticky inspector histogram, Histogram-node
  parity.** The app now starts maximized. The inspector histogram **keeps its
  settings across node switches** — colour-space view, log, smoothing, *and* the
  per-channel toggles + range filters persist (by channel name), so you can compare
  the same curve on the next node. The **Histogram node** gained the inspector's
  controls: a **BGR/HLS** colour space and **smoothing**; its payload now carries
  per-channel names, space, and bin counts (Hue = 180 bins). Suites: 34 smoke + 38
  engine.
- **2026-06-04** — **Auto Cluster peak detection: topographic prominence + histogram
  header polish.** Auto Cluster's **Min peak prominence** now measures each peak
  against its **own surrounding valley** (topographic prominence, relative to the
  peak's height) instead of against the global maximum — so a small colored feature
  on a large uniform background is kept as its own cluster (it was previously
  dropped for not rivalling the background peak), while a bump on the shoulder of a
  dominant peak is still rejected as noise. The inspector histogram header gained
  minimal vertical separators between the view / log / smoothing groups and an
  editable value field beside the smoothing slider. Suites: 33 smoke + 38 engine.
- **2026-06-04** — **Fixed canvas coordinate system + inspector histogram smoothing.**
  The pipeline pane now has a fixed origin pinned at **(0, 0)** top-left; the scene
  only grows right/down to enclose the nodes (it never shifts the origin), so
  zoom/scroll move every node and the grid together and a node's `(x, y)` is a
  stable absolute position. **Loading a pipeline preserves the exact relative
  layout** — older saves that contain negative coordinates are translated into the
  positive quadrant instead of being clamped/cluttered against the edges. The
  inspector **histogram gained a Gaussian smoothing slider** (display only).
  Suites: 33 smoke + 37 engine.
- **2026-06-04** — **Inspector histogram hidden on chart nodes + hue-wrap fix.**
  Nodes whose preview is a plotted graph (the clustering diagnostics and the
  Histogram node) now set `Operation.preview_is_chart=True`, so the inspector pane
  hides its per-channel histogram for them (a histogram of a graph is meaningless).
  Fixed a hue-binning quirk: OpenCV's 8-bit BGR→HLS can emit hue 180, which made
  Auto Cluster's peak histogram a spurious 181 bins; hue now wraps 180→0 (it's
  circular), keeping it a clean 180-bin histogram consistent with the inspector.
- **2026-06-04** — **Parameter tooltips now explain each knob.** Every parameter
  carries a one-line `ParamSpec.help` describing how it affects the result; the
  control's tooltip shows the full name *and* that blurb (was name only). A
  regression test (`test_param_help_present`) enforces that every shown parameter
  documents its effect. Suites: 33 smoke + 37 engine.
- **2026-06-04** — **Clustering preview polish + Auto Cluster saturation control.**
  The diagnostic preview now renders at 1024px (crisp, scaled from the design grid)
  with a shorter proportional palette captioned "palette after clustering". Auto
  Cluster's hue **peak markers are now colored by the real saturation-weighted mean
  BGR** of the pixels at that hue (with a visibility ring), instead of an idealized
  pure hue — so they track the actual image content. Exposed the peak-detection
  **saturation weighting as a parameter** (`sat_weight`, an exponent on `(S/255)`;
  1.0 = the former hardcoded linear weighting, 0 = ignore saturation, >1 = favour
  vivid pixels harder) — gated to peaks mode + the Hue channel via a two-condition
  `enabled_if` (the gray-out mechanism now supports AND-lists). Suites: 33 smoke + 36
  engine.
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
