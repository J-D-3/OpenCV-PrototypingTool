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
  datatypes.py        port data types + connection compatibility rules
  batch.py            Batch: a stack of images flowing on one edge (multi-image)
  graph.py            GraphModel: nodes + edges, the topology source of truth
  engine.py           DAG evaluator: topo order, dirty propagation, caching,
                      error capture, color-space tracking; maps batch elements
                      across a thread pool (parallel fan-out, preview element first)
  codegen.py          walk the graph upstream from a node -> language-neutral
                      pseudocode (Export Code node + Function-info tooltips)
  persistence.py      to_dict / from_dict — save/load a pipeline as JSON

ui/                   frontend — PyQt6 view layer
  controller.py       GraphController: bridges view items <-> model/engine; runs
                      recomputes on a background thread (spinner, coalescing)
  image_utils.py      cv_to_qimage: gray/BGR/BGRA + float normalization for display
  nodes.py            Node / ImageNode / FunctionNode / SaveToFileNode /
                      ExportCodeNode graphics items (+ the recompute spinner)
  node_icons.py       per-operation corner glyphs (hand-drawn, keyed by op id)
  arrow.py            ArrowItem (edge graphics)
  canvas.py           GraphicsImageView + ImageDropWidget (the node canvas)
  viewer.py           ImageViewerWindow (pinned, per-node output inspector)
  inspector_pane.py   InspectorPane: docked live inspector that follows the
                      selection (image + pixel-neighbourhood + histogram filter;
                      shows pseudocode text for the Export Code node)
  parameters.py       ParameterPanel auto-generated from a node's ParamSpec schema
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
3. The dirty nodes are marked "computing" (spinner) and `Engine.evaluate_all()`
   runs **on a background thread** — topological order, caching clean nodes,
   recording any failure on the node (`GraphNode.error`).
4. When the worker finishes, a queued signal (`evalDone`) delivers the result to
   the **GUI thread**, which clears the spinner, refreshes exactly the recomputed
   view items (`refresh_from_model`), and fires `on_commit` side effects (e.g.
   save-to-file) only on committed — not preview — recomputes.

This replaced the old recursive scene-walking + re-entrancy flags, then the
once-synchronous evaluation (see **Concurrency** below).

## Concurrency: responsive UI on large/expensive graphs

Evaluation no longer blocks the UI thread. Two independent layers:

- **Background evaluation** (`ui/controller.py`). Parameter changes *and*
  structural edits (connect / delete / swap / rewire) recompute on a worker
  thread; the worker emits `evalDone` (a `QueuedConnection` signal) so results
  are applied on the GUI thread. Rapid changes **coalesce** — while a run is in
  flight, a new request just sets `_pending` (latest wins) and re-runs when the
  current one finishes. `wait_idle()` pumps the event loop until idle; structural
  ops call it **before mutating topology** (so the worker never sees a concurrent
  structural change), and tests call it to observe results. `recompute_all`
  (load) stays synchronous. Nodes show a gray overlay + animated spinner
  (`Node.set_computing`, driven by a per-node `QTimer`) while recomputing.
- **Parallel batch fan-out** (`core/engine._run_batched`). A node's batch maps
  across a `ThreadPoolExecutor` (`Engine._max_workers = min(8, cpu)`), the
  previewed element first. This parallelises because OpenCV/NumPy release the GIL
  during heavy C work — so threads (no pickling/copying), not processes. The one
  shared global is `cv2.setRNGSeed`, so k-means holds `operations._KMEANS_LOCK`
  across seed+kmeans to stay deterministic under parallel fan-out.

The model stays **node-major** (each node finishes its whole batch before the
next); the previewed element is computed first within a node but is not streamed
end-to-end ahead of the others. Evaluation itself never mutates topology — it
only writes per-node `output`/`dirty`/`error`/`color_space` — which is what makes
the background thread safe alongside main-thread topology reads.

### Internal dependency direction (no cycles)
`operations` ← `nodes` ← `arrow`, `viewer`, `canvas` ← `main_window` ← `app`.
`nodes` references `ArrowItem` only under `TYPE_CHECKING`, so `arrow`↔`nodes`
does not cycle at runtime.

## Where to change things

| I want to… | Go to |
|------------|-------|
| **Add an OpenCV operation** | Add one `Operation` entry in `core/operations.py` (id, label, category, ports, param schema, a `compute(inputs, params)` function). The sidebar and node factory pick it up automatically. |
| Add a many-input node (e.g. Create Batch) | Set `variadic=True` (accepts N inputs) and, to consume batches directly, `raw=True` (engine passes inputs as-is). |
| Give an op a non-image preview (e.g. draw contours) | Set its `render_preview(inputs, output, params) -> image` hook (consumed from Phase 4). |
| Show key stats for an op (e.g. #contours) | Set its `summary(output, params) -> dict` hook (consumed from Phase 4). |
| Give an op a tooltip / better pseudocode | Set `Operation.description` (else its `compute` docstring is used); for a precise code line add a `core/codegen.py` `_CODE` emitter. |
| Make an int slider non-linear | `ParamSpec(kind="int", log=True)` → logarithmic slider (`ui/parameters._add_log_int`). |
| Add a non-image payload op (clusters, contours, regions) | Declare the port `datatypes` type; ops with the same payload type compose (e.g. Label Regions → Filter Contours). |
| Change how a parameter control looks | `ui/parameters.py`; the widget is derived from the `ParamSpec.kind`. |
| Change batch parallelism / worker count | `core/engine.py` (`_max_workers`, `_run_batched`). Hold `operations._KMEANS_LOCK` around any new global cv2 state used inside a parallelised op. |
| Change recompute threading / spinner / coalescing | `ui/controller.py` (`_recompute_async`, `wait_idle`, `evalDone`) + `ui/nodes.Node.set_computing`. |
| Change node appearance / icons | `ui/nodes.py`. |
| Change canvas behaviour (drag, connect, grid) | `ui/canvas.py`. |
| Change the inspector window | `ui/viewer.py`. |
| Change evaluation / propagation | `core/engine.py`. |
| Change how the view drives the backend | `ui/controller.py`. |
| Change the save/load format | `core/persistence.py`. |

## Canvas controls
- **Right-drag** between two nodes: create a connection. Dropping on a full
  single-input node **re-points** (rewires) it; dropping on a node the source is
  **already connected to disconnects** it (toggle — `controller.is_connected` +
  `canvas._disconnect`); connections that would form a cycle are rejected.
- **Double-click** a node: open its inspector.
- **Mouse wheel** over a batch node: scroll through its images (the frame index
  is shared with the inspector pane's slider).
- **Delete / Backspace**: remove the selected node(s) or arrow(s).
- **S**: swap the two inputs of a selected binary op (e.g. Diff A↔B).
- **Save / Load Pipeline** (sidebar): persist the whole graph to JSON.

## Run / test
```powershell
.\.venv\Scripts\Activate.ps1
python main.py [image]                 # launch the GUI
$env:QT_QPA_PLATFORM="offscreen"; python smoke_test.py   # headless checks
```
