"""Headless smoke test / safety net for the node pipeline.

Builds MainWindow offscreen and exercises the real GUI connection path
(GraphicsImageView._create_arrow_between, which also enables downstream
propagation) across:

  * a basic chain            image -> grayscale -> blur
  * a parameter change       propagates downstream (threshold -> blur)
  * a two-input node         sum of two images
  * binary-op input order    diff(A, B) != diff(B, A)
  * save-to-file             writes a file to ./output

Run with QT_QPA_PLATFORM=offscreen (set automatically below).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import json
from dataclasses import replace
import numpy as np
import cv2
from PyQt6 import QtWidgets, QtCore, QtGui

from ui.main_window import MainWindow
from ui.nodes import Node, ImageNode, FunctionNode, SaveToFileNode
from ui.viewer import ImageViewerWindow
from ui.image_utils import cv_to_qimage


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def make_window(app):
    """A window with an empty scene (no initial image node).

    WA_DeleteOnClose makes w.close() tear down the C++ widget tree
    deterministically, so repeatedly creating windows across checks does not
    accumulate half-destroyed Qt objects (which can crash on later GC).
    """
    from PyQt6 import QtCore
    w = MainWindow(None, "smoke")
    w.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    w.show()
    app.processEvents()
    return w


def _new_items(scene, before, cls):
    return [it for it in scene.items()
            if id(it) not in before and isinstance(it, cls)]


def add_image(window, img) -> ImageNode:
    scene = window.drop_widget.view._scene
    before = {id(it) for it in scene.items()}
    window.drop_widget.add_icon(img)
    new = _new_items(scene, before, ImageNode)
    assert new, "no ImageNode created"
    return new[0]


def add_func(window, label) -> FunctionNode:
    scene = window.drop_widget.view._scene
    before = {id(it) for it in scene.items()}
    window.drop_widget.add_function_node(label)
    new = _new_items(scene, before, FunctionNode)
    assert new, f"no FunctionNode created for {label!r}"
    return new[0]


def connect(window, src: Node, dst: Node) -> None:
    """Connect via the real GUI path (creates an ArrowItem + registers input)."""
    window.drop_widget.view._create_arrow_between(src, dst)
    window.drop_widget.view.controller.wait_idle()   # structural edits recompute async


def gradient_bgr(h=120, w=160) -> np.ndarray:
    row = np.linspace(0, 255, w, dtype=np.uint8)
    gray = np.tile(row, (h, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def check_basic_chain(app) -> None:
    w = make_window(app)
    img = gradient_bgr()
    src = add_image(w, img)
    assert src.get_output_image() is not None

    gray = add_func(w, "To Grayscale")
    connect(w, src, gray)
    app.processEvents()
    out = gray.get_output_image()
    assert out is not None and out.ndim == 2, f"grayscale bad output: {None if out is None else out.shape}"

    blur = add_func(w, "Blur")
    connect(w, gray, blur)
    app.processEvents()
    assert blur.get_output_image() is not None, "blur produced no output"
    w.close()
    print("OK  basic chain: image -> grayscale -> blur")


def check_param_propagation(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    thresh = add_func(w, "Threshold")
    blur = add_func(w, "Blur")
    connect(w, src, thresh)
    connect(w, thresh, blur)
    app.processEvents()

    thresh.set_parameter("threshold_value", 50, preview_mode=False)
    thresh.controller.wait_idle()   # param eval runs on a background thread
    blur_low = blur.get_output_image()
    assert blur_low is not None, "no downstream output after first param set"
    blur_low = blur_low.copy()

    thresh.set_parameter("threshold_value", 200, preview_mode=False)
    thresh.controller.wait_idle()
    blur_high = blur.get_output_image()
    assert blur_high is not None

    assert not np.array_equal(blur_low, blur_high), \
        "downstream (blur) output did not change after upstream threshold param change"
    w.close()
    print("OK  parameter change propagates downstream (threshold -> blur)")


def check_two_input_sum(app) -> None:
    w = make_window(app)
    a = add_image(w, np.full((100, 120, 3), 40, np.uint8))
    b = add_image(w, np.full((100, 120, 3), 200, np.uint8))
    s = add_func(w, "Sum")
    assert s.op.id == "sum"
    connect(w, a, s)
    app.processEvents()
    assert s.get_output_image() is None, "Sum executed with only one input"
    connect(w, b, s)
    app.processEvents()
    out = s.get_output_image()
    assert out is not None and out.shape == (100, 120, 3), "Sum produced no/badly-shaped output"
    w.close()
    print("OK  two-input node: Sum executes only once both inputs connected")


def check_diff_input_order(app) -> None:
    imgA = np.full((80, 90, 3), 200, np.uint8)
    imgB = np.full((80, 90, 3), 50, np.uint8)

    # diff(A, B): 200 - 50 = 150
    w1 = make_window(app)
    a1, b1 = add_image(w1, imgA), add_image(w1, imgB)
    d1 = add_func(w1, "Diff")
    assert d1.op.id == "diff"
    connect(w1, a1, d1)
    connect(w1, b1, d1)
    app.processEvents()
    ab = d1.get_output_image()

    # diff(B, A): 50 - 200 -> saturates to 0
    w2 = make_window(app)
    a2, b2 = add_image(w2, imgA), add_image(w2, imgB)
    d2 = add_func(w2, "Diff")
    connect(w2, b2, d2)
    connect(w2, a2, d2)
    app.processEvents()
    ba = d2.get_output_image()

    assert ab is not None and ba is not None, "Diff produced no output"
    assert not np.array_equal(ab, ba), "diff(A,B) should differ from diff(B,A)"
    assert int(ab.mean()) > 100 and int(ba.mean()) < 10, \
        f"unexpected diff values: mean(A-B)={ab.mean():.0f}, mean(B-A)={ba.mean():.0f}"
    w1.close()
    w2.close()
    print("OK  binary-op input order respected: diff(A,B) != diff(B,A)")


def check_save_to_file(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    save = add_func(w, "Save to File")
    assert isinstance(save, SaveToFileNode)

    fname = "smoke_save_test_DELETEME.png"
    out_path = os.path.join("output", fname)
    if os.path.exists(out_path):
        os.remove(out_path)

    save.set_parameter("use_custom", True)
    save.set_parameter("filename", fname)
    connect(w, src, save)          # connecting triggers execution -> write
    app.processEvents()

    assert os.path.exists(out_path), f"save-to-file did not write {out_path}"
    os.remove(out_path)
    w.close()
    print("OK  save-to-file wrote and cleaned up output/" + fname)


def check_inspector(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    thresh = add_func(w, "Threshold")     # an op with parameters
    connect(w, src, thresh)
    app.processEvents()
    viewer = ImageViewerWindow(thresh)
    viewer.show()
    app.processEvents()
    assert not viewer._image._pixmap.isNull(), "inspector showed no image"
    assert "×" in viewer._meta.text(), "inspector should show size metadata"
    assert not hasattr(viewer, "_params"), "params are edited only in the main window panel"
    viewer._on_hover(3, 4)
    assert "x=3 y=4" in viewer._readout.text(), "inspector should show the pixel readout"
    viewer.close()
    w.close()
    print("OK  dedicated inspector: image, metadata, pixel readout")


def check_parameter_panel(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    thresh = add_func(w, "Threshold")
    connect(w, src, thresh)
    app.processEvents()

    scene = w.drop_widget.view._scene
    thresh.setSelected(True)        # fires the scene selection handler
    app.processEvents()
    assert w.param_panel.has_controls(), "threshold should expose auto-built controls"

    # Log-scaled int sliders (Filter Contours area params): build an isolated
    # panel so stale deleteLater widgets don't pollute the child lookup.
    from ui.parameters import ParameterPanel
    cf = add_func(w, "Filter Contours")
    panel = ParameterPanel()
    panel.set_node(cf)
    sliders = panel.findChildren(QtWidgets.QSlider)
    assert len(sliders) == 2, f"expected 2 area sliders, got {len(sliders)}"
    assert all(s.maximum() == 1000 for s in sliders), "log sliders use a 0..1000 position range"
    fields = [f.text() for f in panel.findChildren(QtWidgets.QLineEdit)]
    assert "100.000" in fields, f"max_area value field should show dotted default, got {fields}"
    panel.deleteLater()

    # An op with no parameters should produce no controls.
    scene.clearSelection()
    gray = add_func(w, "To Grayscale")
    gray.setSelected(True)
    app.processEvents()
    assert not w.param_panel.has_controls(), "no-param op should expose no controls"
    w.close()
    print("OK  parameter panel auto-builds controls from the op schema")


def check_display_conversion(app) -> None:
    # grayscale (single channel)
    q = cv_to_qimage(np.zeros((10, 12), np.uint8))
    assert not q.isNull() and (q.width(), q.height()) == (12, 10)
    # float (e.g. Fourier magnitude) -> normalized, no crash
    q = cv_to_qimage(np.random.rand(8, 8).astype(np.float32) * 1000)
    assert not q.isNull()
    # BGR
    q = cv_to_qimage(np.zeros((6, 7, 3), np.uint8))
    assert (q.width(), q.height()) == (7, 6)
    print("OK  cv_to_qimage handles gray / float / bgr")


def check_preview_and_summary(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    thresh = add_func(w, "Threshold")
    blur = add_func(w, "Blur")
    connect(w, src, thresh)
    connect(w, thresh, blur)
    app.processEvents()

    # Stub render_preview + summary on blur's op (does not touch the registry).
    blur.op = replace(
        blur.op,
        render_preview=lambda inputs, out, params: np.full_like(out, 7),
        summary=lambda out, params: {"pixels": int(out.size)},
    )
    preview = blur.get_preview_image()
    assert preview is not None and int(preview.flat[0]) == 7, "render_preview not used"
    assert blur.get_summary().get("pixels") == blur.get_output_image().size

    # Signal-driven: an upstream change fires nodeChanged for the downstream node.
    changed = []
    blur.controller.signals.nodeChanged.connect(changed.append)
    thresh.set_parameter("threshold_value", 99, preview_mode=False)
    thresh.controller.wait_idle()
    assert blur in changed, "downstream node change did not signal"

    viewer = ImageViewerWindow(blur)
    viewer.show()
    app.processEvents()
    assert viewer._summary.isVisible() and "pixels" in viewer._summary.text()
    viewer.close()
    w.close()
    print("OK  inspector is signal-driven and uses render_preview + summary")


def check_save_load(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    thresh = add_func(w, "Threshold")
    connect(w, src, thresh)
    app.processEvents()
    expected = thresh.get_output_image().copy()

    data = json.loads(json.dumps(w.drop_widget.to_dict()))  # round-trip through JSON
    w.drop_widget.load_dict(data)
    app.processEvents()

    scene = w.drop_widget.view._scene
    funcs = [it for it in scene.items() if isinstance(it, FunctionNode)]
    imgs = [it for it in scene.items() if isinstance(it, ImageNode)]
    assert len(funcs) == 1 and len(imgs) == 1, "node count not preserved across save/load"
    assert np.array_equal(expected, funcs[0].get_output_image()), "result not preserved"
    w.close()
    print("OK  pipeline save/load round-trips structure and result")


def check_progressive_load(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    g = add_func(w, "To Grayscale")
    b = add_func(w, "Blur")
    connect(w, src, g)
    connect(w, g, b)
    app.processEvents()
    data = json.loads(json.dumps(w.drop_widget.to_dict()))

    ctrl = w.drop_widget.view.controller
    reveals = []
    ctrl.signals.nodeChanged.connect(reveals.append)   # one emit per node as it finishes

    w.drop_widget.load_dict(data)                       # synchronous but progressive
    app.processEvents()

    nodes = list(ctrl.model.nodes.values())
    assert len(reveals) == len(nodes), \
        f"load should reveal each node once ({len(reveals)} vs {len(nodes)})"
    assert all(n.output is not None for n in nodes), "all nodes computed after load"
    assert not any(getattr(qt, "_executing", False) for qt in ctrl._qt_by_gid.values()), \
        "no spinner should remain after load"
    w.close()
    print("OK  progressive load: graph drawn, spinners, per-node reveal")


def check_delete_node(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    blur = add_func(w, "Blur")
    connect(w, src, blur)
    app.processEvents()
    assert blur.get_output_image() is not None

    view = w.drop_widget.view
    view._delete_node(src)
    view.controller.wait_idle()
    assert src not in view._scene.items(), "deleted node still in scene"
    assert blur.get_output_image() is None, "downstream not re-evaluated after delete"
    w.close()
    print("OK  node deletion removes it and re-evaluates downstream")


def check_input_swap(app) -> None:
    w = make_window(app)
    a = add_image(w, np.full((40, 40, 3), 200, np.uint8))
    b = add_image(w, np.full((40, 40, 3), 50, np.uint8))
    diff = add_func(w, "Diff")
    connect(w, a, diff)
    connect(w, b, diff)
    app.processEvents()
    before = diff.get_output_image().copy()      # 200-50 = 150
    assert w.drop_widget.view.controller.swap_inputs(diff)
    w.drop_widget.view.controller.wait_idle()
    after = diff.get_output_image()               # 50-200 -> 0
    assert not np.array_equal(before, after), "input swap did not change result"
    w.close()
    print("OK  binary-op input swap reverses the operands")


def check_color_chain(app) -> None:
    # The user's target chain: Load > To HSL > Cluster > Reduce Colors.
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    hls = add_func(w, "To HSL")
    km = add_func(w, "K-Means Cluster")
    red = add_func(w, "Reduce Colors")
    connect(w, src, hls)
    connect(w, hls, km)
    connect(w, km, red)        # CLUSTERS->CLUSTERS connection (type-validated)
    app.processEvents()

    assert isinstance(km.get_output_image(), dict), "cluster node should output a clusters payload"
    assert km.get_summary().get("clusters") == 6, "cluster summary should report k"
    assert isinstance(km.get_preview_image(), np.ndarray), "cluster preview should be a swatch image"

    out = red.get_output_image()
    assert isinstance(out, np.ndarray) and out.shape == gradient_bgr().shape, "reduce should output an image"
    uniq = np.unique(out.reshape(-1, 3), axis=0)
    assert uniq.shape[0] <= 6, f"reduced image should have <= 6 colors, got {uniq.shape[0]}"
    w.close()
    print("OK  Load > To HSL > K-Means > Reduce Colors chain runs")


def check_segmentation_chain(app) -> None:
    # Resize > Blur > Adaptive Threshold > Find Contours > Filter Contours.
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    chain = [add_func(w, n) for n in
             ("Resize", "Blur", "Adaptive Threshold", "Find Contours", "Filter Contours")]
    prev = src
    for node in chain:
        connect(w, prev, node)
        prev = node
    app.processEvents()

    find, filt = chain[3], chain[4]
    fc_out = find.get_output_image()
    assert isinstance(fc_out, dict) and "contours" in fc_out, "find_contours should output a contours payload"
    assert isinstance(find.get_preview_image(), np.ndarray), "contours preview should be a drawn image"
    assert "contours" in find.get_summary()
    flt_out = filt.get_output_image()
    assert isinstance(flt_out, dict) and len(flt_out["contours"]) <= len(fc_out["contours"])
    assert "kept" in filt.get_summary()
    w.close()
    print("OK  Resize > Blur > Adaptive Threshold > Find Contours > Filter chain runs")


def check_fourier_chain(app) -> None:
    # Load > DFT > Inverse DFT, and verify the reconstruction matches the input.
    w = make_window(app)
    img = gradient_bgr()
    src = add_image(w, img)
    dft = add_func(w, "DFT")
    idft = add_func(w, "Inverse DFT")
    connect(w, src, dft)
    connect(w, dft, idft)
    app.processEvents()

    assert isinstance(dft.get_output_image(), dict), "DFT should output a spectrum payload"
    assert isinstance(dft.get_preview_image(), np.ndarray), "DFT preview should be a magnitude image"
    back = idft.get_output_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    assert back is not None and np.allclose(back, gray, atol=1e-2), "idft(dft(img)) != img"
    w.close()
    print("OK  Load > DFT > Inverse DFT reconstructs the image")


def check_rewire(app) -> None:
    w = make_window(app)
    a = add_image(w, np.full((30, 30, 3), 10, np.uint8))
    b = add_image(w, np.full((30, 30, 3), 200, np.uint8))
    blur = add_func(w, "Blur")
    connect(w, a, blur)
    app.processEvents()
    assert int(blur.get_output_image().mean()) < 50, "blur should reflect source A (~10)"

    # Drag B onto the already-connected single-input Blur -> rewire.
    connect(w, b, blur)
    app.processEvents()
    model = w.drop_widget.view.controller.model
    assert len(model.incoming(blur.gnode)) == 1, "rewire should not add a second input"
    assert int(blur.get_output_image().mean()) > 150, "blur should now reflect source B (~200)"
    w.close()
    print("OK  drag-to-rewire repoints a full single-input connection")


def check_inspector_pane(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    thresh = add_func(w, "Threshold")
    connect(w, src, thresh)
    app.processEvents()

    pane = w.inspector_pane
    # Selecting a node populates the pane: image, histogram channels, base patch source.
    thresh.setSelected(True)
    app.processEvents()
    assert pane._node is thresh
    assert pane._disp is not None and pane._disp.ndim == 2, "threshold output should be single-channel"
    assert len(pane._hist._channels) == 1, "single-channel image -> one histogram channel"
    assert "×" in pane._meta.text() and "Gray" in pane._meta.text(), "metadata should show size + type"
    assert pane._neigh.grid_size() == 27, "default grid should be 27x27"

    # Hover updates the neighbourhood; left-click freezes; right-click releases.
    pane._on_hover(5, 4)
    assert pane._neigh._center == (5, 4)
    pane._on_click(7, 8)
    assert pane._frozen and pane._neigh._center == (7, 8)
    pane._on_hover(1, 1)
    assert pane._neigh._center == (7, 8), "frozen neighbourhood should ignore hover"
    pane._on_release(3, 3)
    assert not pane._frozen and pane._neigh._center == (3, 3), "right-click should release"
    pane._on_hover(2, 2)
    assert pane._neigh._center == (2, 2), "hover should resume after release"

    # Regression: a centre from a larger frame must not crash on a smaller image
    # (batches can mix image sizes). The centre is clamped to the new bounds.
    pane._neigh._center = (999, 999)
    pane._neigh.set_base(np.zeros((8, 12, 3), np.uint8), ["B", "G", "R"])
    assert "x=11 y=7" in pane._neigh._readout.text(), "out-of-bounds centre should clamp, not crash"

    # Narrowing a histogram range masks the preview AND the neighbourhood grid.
    base_nonzero = int(np.count_nonzero(pane._disp))
    pane._hist._channels[0]["slider"]._lo = 0
    pane._hist._channels[0]["slider"]._hi = 100   # excludes the white (255) pixels
    pane._apply_filter()
    assert not pane._image._pixmap.isNull(), "filtered image should still render"
    assert int(np.count_nonzero(pane._neigh._base)) < base_nonzero, \
        "neighbourhood grid should reflect the histogram filter"
    # Restore full range for the rest of the test.
    pane._hist._channels[0]["slider"]._lo = 0
    pane._hist._channels[0]["slider"]._hi = 255
    pane._apply_filter()

    # Name-based curve colors: Gray draws dark gray.
    from ui.inspector_pane import _channel_color
    assert pane._hist._channels[0]["color"] == _channel_color("Gray")

    # Log-scale toggle recomputes the plot without error.
    pane._hist._log_cb.setChecked(True)
    app.processEvents()

    # A 3-channel image gives three histogram channels labelled B/G/R, and the
    # old single 'Gray' row must be cleared (no leftover widgets).
    img_node = add_image(w, gradient_bgr())
    w.drop_widget.view._scene.clearSelection()
    img_node.setSelected(True)
    app.processEvents()
    assert len(pane._hist._channels) == 3
    assert [c["name"] for c in pane._hist._channels] == ["B", "G", "R"]
    assert pane._hist._rows.count() == 3, "stale channel rows were not cleared"

    # Wheel zoom keeps the image point under the cursor anchored.
    ip = pane._image
    ip.resize(200, 160)
    ip._fit()
    base_scale = ip._scale
    cursor = QtCore.QPointF(120.0, 90.0)
    img_pt_before = ((cursor.x() - ip._origin.x()) / ip._scale,
                     (cursor.y() - ip._origin.y()) / ip._scale)
    ip._zoom_at(cursor, 2.0)
    assert ip._scale > base_scale, "zoom-in should increase scale"
    img_pt_after = ((cursor.x() - ip._origin.x()) / ip._scale,
                    (cursor.y() - ip._origin.y()) / ip._scale)
    assert abs(img_pt_before[0] - img_pt_after[0]) < 1e-6, "zoom must anchor to the cursor"
    w.close()
    print("OK  inspector pane: colors, channel-clear, log toggle, zoom-to-cursor")


def check_batch(app) -> None:
    import os
    import glob
    from core.batch import Batch

    w = make_window(app)
    imgs = [np.full((20, 24, 3), v, np.uint8) for v in (20, 120, 220)]
    src = w.drop_widget.add_images(imgs)          # one batch source of 3 images
    blur = add_func(w, "Blur")
    save = add_func(w, "Save to File")

    pattern = os.path.join("output", "batch_smoke_DELETEME*")
    for f in glob.glob(pattern):
        os.remove(f)
    save.set_parameter("use_custom", True)
    save.set_parameter("filename", "batch_smoke_DELETEME.png")

    connect(w, src, blur)
    connect(w, blur, save)                        # commit -> save writes every element
    app.processEvents()

    # The chain ran once but produced a batch of 3 results.
    assert isinstance(blur.gnode.output, Batch) and len(blur.gnode.output) == 3

    # The frame index selects which element every node previews.
    ctrl = w.drop_widget.view.controller
    ctrl.set_preview_index(0)
    assert int(blur.get_output_image().mean()) == 20
    ctrl.set_preview_index(2)
    assert int(blur.get_output_image().mean()) == 220

    # Save-to-File wrote one file per image.
    files = glob.glob(pattern)
    assert len(files) == 3, f"expected 3 saved files, got {len(files)}"
    for f in files:
        os.remove(f)

    # Selecting the batch source shows the frame nav "< i/3 >".
    src.setSelected(True)
    app.processEvents()
    assert w.inspector_pane._frame_nav.isVisible()
    assert w.inspector_pane._frame_label.text().endswith("/3")
    w.close()
    print("OK  batched: one chain over 3 images; per-frame preview + save-all")


def check_create_batch(app) -> None:
    from core.batch import Batch

    w = make_window(app)
    a = add_image(w, np.full((16, 16, 3), 30, np.uint8))
    b = add_image(w, np.full((16, 16, 3), 130, np.uint8))
    c = add_image(w, np.full((16, 16, 3), 230, np.uint8))
    cb = add_func(w, "Create Batch")
    blur = add_func(w, "Blur")
    connect(w, a, cb)
    connect(w, b, cb)
    connect(w, c, cb)              # three inputs into one variadic node
    connect(w, cb, blur)
    app.processEvents()

    assert isinstance(cb.gnode.output, Batch) and len(cb.gnode.output) == 3
    assert isinstance(blur.gnode.output, Batch) and len(blur.gnode.output) == 3

    ctrl = w.drop_widget.view.controller
    ctrl.set_preview_index(0)
    assert int(blur.get_output_image().mean()) == 30
    ctrl.set_preview_index(2)
    assert int(blur.get_output_image().mean()) == 230

    # Variadic node keeps accepting more inputs.
    d = add_image(w, np.zeros((16, 16, 3), np.uint8))
    assert cb.can_accept_input(d)
    w.close()
    print("OK  Create Batch: variadic inputs -> one batch through the chain")


def check_node_icons_and_scroll(app) -> None:
    import core.operations as ops
    from ui import node_icons

    # Every operation's glyph draws without error.
    pm = QtGui.QPixmap(16, 16)
    for op_id in ops.REGISTRY:
        p = QtGui.QPainter(pm)
        node_icons.draw(p, QtCore.QRectF(0, 0, 16, 16), op_id, QtGui.QColor(50, 50, 50))
        p.end()

    w = make_window(app)
    # A freshly added function node shows its rendered icon immediately, not the
    # gray placeholder (sample a white interior pixel away from text/border).
    f = add_func(w, "Blur")
    size = f.pixmap().width()
    col = f.pixmap().toImage().pixelColor(size // 2, size - 6)
    # Rendered function icon has a light-green background (200,255,200); the
    # unrendered placeholder is gray (200,200,200) — distinguish by the green.
    assert col.green() > 240 and col.red() < 230, \
        "function node should render its green icon immediately, not the gray placeholder"

    # Mouse-wheel scrolls a batch node's previewed element (clamped).
    src = w.drop_widget.add_images([np.full((10, 10, 3), v, np.uint8) for v in (10, 100, 200)])
    view = w.drop_widget.view
    assert view._scroll_batch(src, 1) and view.controller.preview_index == 1
    view._scroll_batch(src, 9)
    assert view.controller.preview_index == 2
    assert view._scroll_batch(f, 1) is False, "non-batch node should not scroll"
    w.close()
    print("OK  node glyphs draw; immediate label; batch wheel-scroll")


def check_icon_size_control(app) -> None:
    w = make_window(app)
    dw = w.drop_widget
    assert dw.icon_size == 90, "default icon size should be 90"
    f = add_func(w, "Blur")
    assert f.pixmap().width() == 90
    dw._size_slider.setValue(140)
    app.processEvents()
    assert dw.icon_size == 140
    assert f.pixmap().width() == 140, "existing nodes should resize with the slider"
    w.close()
    print("OK  canvas icon-size control: default 90, resizes nodes")


def check_save_nonimage(app) -> None:
    import os
    import glob

    w = make_window(app)
    img = np.zeros((40, 40, 3), np.uint8)
    cv2.rectangle(img, (5, 5), (22, 22), (255, 255, 255), -1)
    src = add_image(w, img)
    fc = add_func(w, "Find Contours")
    save = add_func(w, "Save to File")

    pattern = os.path.join("output", "nonimg_DELETEME*")
    for f in glob.glob(pattern):
        os.remove(f)
    save.set_parameter("use_custom", True)
    save.set_parameter("filename", "nonimg_DELETEME.png")

    connect(w, src, fc)
    # CONTOURS -> Save (input type ANY) must be allowed and save the rendered preview.
    assert save.can_accept_input(fc), "Save to File should accept a non-image (contours) output"
    connect(w, fc, save)
    app.processEvents()

    files = glob.glob(pattern)
    assert len(files) == 1, f"expected the contours preview to be saved, got {len(files)}"
    loaded = cv2.imread(files[0])
    assert loaded is not None and loaded.ndim == 3, "saved fallback should be a valid image"
    for f in files:
        os.remove(f)
    w.close()
    print("OK  save-to-file falls back to the display image (e.g. contours)")


def check_disconnect(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    blur = add_func(w, "Blur")
    view = w.drop_widget.view
    model = view.controller.model

    connect(w, src, blur)
    app.processEvents()
    assert view.controller.is_connected(src, blur)
    assert blur.get_output_image() is not None

    # Right-dragging the same source onto the connected target disconnects it.
    connect(w, src, blur)
    app.processEvents()
    assert not view.controller.is_connected(src, blur), "second drag should disconnect"
    assert len(model.incoming(blur.gnode)) == 0
    assert blur.get_output_image() is None, "downstream should re-evaluate after disconnect"
    # the arrow is gone from the scene
    from ui.arrow import ArrowItem
    assert not any(isinstance(it, ArrowItem) for it in view._scene.items()), "arrow not removed"
    w.close()
    print("OK  drag-to-toggle disconnects an existing connection")


def check_export_code(app) -> None:
    import os
    import glob
    from ui.nodes import ExportCodeNode

    w = make_window(app)
    src = add_image(w, gradient_bgr())
    gray = add_func(w, "To Grayscale")
    exp = add_func(w, "Export Code")
    assert isinstance(exp, ExportCodeNode), "Export Code should build an ExportCodeNode"
    connect(w, src, gray)
    connect(w, gray, exp)
    app.processEvents()

    # Pseudocode reflects the upstream chain.
    code = exp.get_pseudocode()
    assert "imread(" in code and "cvtColor" in code, f"codegen missing expected calls:\n{code}"

    # Inspector pane shows the code text for this node (and hides it otherwise).
    w.inspector_pane.set_node(exp)
    app.processEvents()
    assert w.inspector_pane._code.isVisible(), "inspector should show the code panel for Export Code"
    assert "imread(" in w.inspector_pane._code.toPlainText()
    w.inspector_pane.set_node(gray)
    app.processEvents()
    assert not w.inspector_pane._code.isVisible(), "code panel should hide for ordinary nodes"

    # on_commit writes the pseudocode to ./output.
    if not hasattr(exp, "_node_index"):
        exp._node_index = id(exp) % 10000
    path = os.path.join("output", f"pipeline_{exp._node_index}.txt")
    if os.path.exists(path):
        os.remove(path)
    exp.on_commit()
    assert os.path.exists(path), f"export code did not write {path}"
    os.remove(path)
    w.close()
    print("OK  export code: upstream pseudocode in inspector + written to ./output")


def check_function_search(app) -> None:
    w = make_window(app)
    tree, search = w.func_tree, w.func_search

    def visible_ops():
        out = []
        for i in range(tree.topLevelItemCount()):
            c = tree.topLevelItem(i)
            if c.isHidden():
                continue
            for j in range(c.childCount()):
                it = c.child(j)
                if not it.isHidden():
                    out.append(it.text(0))
        return out

    total = len(visible_ops())
    assert total > 10, "tree should list many ops"

    # By a cv:: call: matches Gaussian Blur *and* every op that calls it
    # internally (Auto Cluster smooths the histogram; Local HDR's low-pass).
    search.setText("gaussianblur")
    vis = visible_ops()
    assert "Gaussian Blur" in vis and "Auto Cluster" in vis, vis

    search.setText("kmeans")                  # cv::kmeans -> both clustering ops
    assert set(visible_ops()) == {"K-Means Cluster", "Auto Cluster"}, visible_ops()

    search.setText("contours")                # by category -> the whole category
    vis = visible_ops()
    assert "Find Contours" in vis and "Filter Contours" in vis

    # An op is findable by a cv:: call it makes *internally* on any code path:
    # Flood Fill uses cv::connectedComponents in its delta==0 branch.
    search.setText("connectedcomponents")
    vis = visible_ops()
    assert "Flood Fill" in vis and "Connected Components" in vis, vis

    search.setText("zzz_nomatch")
    assert visible_ops() == []

    search.setText("")                        # cleared restores everything
    assert len(visible_ops()) == total

    ig = [g for g in w.findChildren(QtWidgets.QGroupBox) if g.title() == "Function info"][0]
    assert ig.minimumHeight() == 200 and ig.maximumHeight() == 200, "info panel should be fixed 200px"
    w.close()
    print("OK  function search filters by name/category/cv:: call; info panel fixed 200px")


def check_live_slider(app) -> None:
    from ui.parameters import ParameterPanel
    from core.operations import REGISTRY
    # Filter Contours area params are 'live'; a normal Blur slider is not.
    cf = REGISTRY["contour_filter"]
    assert all(p.live for p in cf.params if p.name in ("min_area", "max_area"))
    assert not REGISTRY["blur"].params[0].live

    w = make_window(app)
    node = add_func(w, "Filter Contours")
    rec = []
    node.set_parameter = lambda name, value, preview_mode=False: rec.append((name, value))
    panel = ParameterPanel()
    panel.set_node(node)
    s = panel.findChildren(QtWidgets.QSlider)[0]
    s.setSliderDown(True)                 # simulate dragging
    s.setValue(s.value() + 7)
    assert rec, "a 'live' slider should commit while being dragged"
    w.close()
    print("OK  live slider: Filter Contours area evaluates while dragging")


def check_canvas_zoom_scroll(app) -> None:
    w = make_window(app)
    view = w.drop_widget.view
    vp = view.viewport().rect()
    sr = view._scene.sceneRect()
    assert sr.width() >= vp.width() * 1.8 and sr.height() >= vp.height() * 1.8, \
        "scene should be ~2x the viewport so large pipelines have room to scroll"
    assert view.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
    z0 = view._zoom_level
    view._zoom(1.15)
    assert view._zoom_level > z0, "Ctrl+wheel zoom-in raises the zoom level"
    for _ in range(40):
        view._zoom(0.5)
    assert view._zoom_level >= 0.3, "zoom-out is clamped"
    w.close()
    print("OK  canvas: 2x scrollable scene + clamped zoom")


def check_flow_highlight(app) -> None:
    from ui.arrow import ArrowItem
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    g = add_func(w, "To Grayscale")
    b = add_func(w, "Blur")
    side = add_func(w, "Invert")
    connect(w, src, g)
    connect(w, g, b)
    connect(w, src, side)            # a sibling branch off the same source

    w.drop_widget.view._scene.clearSelection()
    g.setSelected(True)
    app.processEvents()
    assert g._flow_role == "selected", "selected node should be yellow"
    assert src._flow_role == "flow" and b._flow_role == "flow", "up/downstream go green"
    assert side._flow_role is None, "a sibling branch is not on the selected flow"

    arrows = {(it.a, it.b): it for it in w.drop_widget.view._scene.items()
              if isinstance(it, ArrowItem)}
    assert arrows[(src, g)]._flow and arrows[(g, b)]._flow, "flow edges go green"
    assert not arrows[(src, side)]._flow, "off-flow edge stays black"

    assert isinstance(b.gnode.comp_time_ms, float), "function node records a compute time"

    g.setSelected(False)
    app.processEvents()
    assert g._flow_role is None and src._flow_role is None, "highlight clears on deselect"
    w.close()
    print("OK  flow highlight: selected=yellow, predecessors/successors=green; comp time tracked")


def check_background_eval(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    thresh = add_func(w, "Threshold")
    blur = add_func(w, "Blur")
    connect(w, src, thresh)
    connect(w, thresh, blur)
    ctrl = thresh.controller

    # A param change recomputes off the UI thread; affected nodes show the
    # spinner immediately and the eval has not completed yet (no event pumped).
    thresh.set_parameter("threshold_value", 30, preview_mode=False)
    assert ctrl._busy, "param change should start a background eval"
    assert thresh._executing and blur._executing, "recomputing nodes should show the spinner"
    ctrl.wait_idle()
    assert not ctrl._busy and not thresh._executing and not blur._executing, \
        "spinner/busy should clear once the background eval finishes"
    out30 = blur.get_output_image().copy()

    # Rapid changes coalesce (latest wins) without piling up threads or crashing.
    for v in (60, 120, 200):
        thresh.set_parameter("threshold_value", v, preview_mode=False)
    ctrl.wait_idle()
    out200 = blur.get_output_image()
    assert not np.array_equal(out30, out200), "coalesced edits should reach the latest value"

    # Structural edits (connect/delete) recompute on the background thread too.
    inv = add_func(w, "Invert")
    w.drop_widget.view._create_arrow_between(blur, inv)   # raw connect (no wait)
    assert ctrl._busy or inv._executing, "connect should trigger a background recompute"
    ctrl.wait_idle()
    assert inv.get_output_image() is not None, "downstream of an async connect should compute"
    w.close()
    print("OK  background eval: off-thread recompute (param + connect) + spinner + coalescing")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    checks = [
        check_basic_chain,
        check_param_propagation,
        check_two_input_sum,
        check_diff_input_order,
        check_save_to_file,
        check_inspector,
        check_parameter_panel,
        check_display_conversion,
        check_preview_and_summary,
        check_save_load,
        check_progressive_load,
        check_delete_node,
        check_input_swap,
        check_color_chain,
        check_segmentation_chain,
        check_fourier_chain,
        check_rewire,
        check_disconnect,
        check_inspector_pane,
        check_batch,
        check_create_batch,
        check_node_icons_and_scroll,
        check_icon_size_control,
        check_save_nonimage,
        check_export_code,
        check_function_search,
        check_live_slider,
        check_canvas_zoom_scroll,
        check_flow_highlight,
        check_background_eval,
    ]
    for chk in checks:
        chk(app)
        app.processEvents()   # let WA_DeleteOnClose tear closed windows down now
    print(f"\nSMOKE OK: {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
