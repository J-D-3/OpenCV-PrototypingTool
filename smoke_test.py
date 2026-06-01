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
import numpy as np
import cv2
from PyQt6 import QtWidgets

from ui.main_window import MainWindow
from ui.nodes import Node, ImageNode, FunctionNode, SaveToFileNode
from ui.viewer import ImageViewerWindow


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def make_window(app) -> MainWindow:
    """A window with an empty scene (no initial image node)."""
    w = MainWindow(None, "smoke")
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


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    checks = [
        check_basic_chain,
        check_param_propagation,
        check_two_input_sum,
        check_diff_input_order,
        check_save_to_file,
        check_inspector,
    ]
    for chk in checks:
        chk(app)
    print(f"\nSMOKE OK: {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
