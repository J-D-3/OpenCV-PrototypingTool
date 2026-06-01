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
from PyQt6 import QtWidgets, QtCore

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
    app.processEvents()
    blur_low = blur.get_output_image()
    assert blur_low is not None, "no downstream output after first param set"
    blur_low = blur_low.copy()

    thresh.set_parameter("threshold_value", 200, preview_mode=False)
    app.processEvents()
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
    gray = add_func(w, "To Grayscale")
    connect(w, src, gray)
    app.processEvents()
    viewer = ImageViewerWindow(gray)
    viewer.show()
    app.processEvents()
    assert not viewer.image_label.pixmap().isNull(), "inspector showed no pixmap"
    viewer.close()
    w.close()
    print("OK  inspector window renders a node's output")


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
    app.processEvents()
    assert blur in changed, "downstream node change did not signal"

    viewer = ImageViewerWindow(blur)
    viewer.show()
    app.processEvents()
    assert viewer.summary_label.isVisible() and "pixels" in viewer.summary_label.text()
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


def check_delete_node(app) -> None:
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    blur = add_func(w, "Blur")
    connect(w, src, blur)
    app.processEvents()
    assert blur.get_output_image() is not None

    view = w.drop_widget.view
    view._delete_node(src)
    app.processEvents()
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
    app.processEvents()
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
    # Shrink > Blur > Adaptive Threshold > Find Contours > Filter Contours.
    w = make_window(app)
    src = add_image(w, gradient_bgr())
    chain = [add_func(w, n) for n in
             ("Shrink", "Blur", "Adaptive Threshold", "Find Contours", "Filter Contours")]
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
    print("OK  Shrink > Blur > Adaptive Threshold > Find Contours > Filter chain runs")


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

    # Hover updates the neighbourhood readout; click freezes it.
    pane._on_hover(5, 4)
    assert pane._neigh._center == (5, 4)
    pane._on_click(7, 8)
    assert pane._frozen and pane._neigh._center == (7, 8)
    pane._on_hover(1, 1)
    assert pane._neigh._center == (7, 8), "frozen neighbourhood should ignore hover"

    # Narrowing a histogram range masks the preview (fewer non-zero pixels).
    base_nonzero = int(np.count_nonzero(pane._disp))
    pane._hist._channels[0]["slider"]._lo = 200
    pane._hist._channels[0]["slider"]._hi = 255
    pane._apply_filter()
    shown = pane._image._pixmap
    assert not shown.isNull(), "filtered image should still render"

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

    # Selecting the batch source shows the frame slider (range 0..2).
    src.setSelected(True)
    app.processEvents()
    assert w.inspector_pane._frame_widget.isVisible()
    assert w.inspector_pane._frame_slider.maximum() == 2
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
        check_delete_node,
        check_input_swap,
        check_color_chain,
        check_segmentation_chain,
        check_fourier_chain,
        check_rewire,
        check_inspector_pane,
        check_batch,
        check_create_batch,
    ]
    for chk in checks:
        chk(app)
        app.processEvents()   # let WA_DeleteOnClose tear closed windows down now
    print(f"\nSMOKE OK: {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
