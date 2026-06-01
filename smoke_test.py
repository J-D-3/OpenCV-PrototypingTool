"""Headless smoke test: construct the GUI offscreen, wire a small pipeline,
and confirm nodes execute. Run with QT_QPA_PLATFORM=offscreen."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import numpy as np
from PyQt6 import QtWidgets

from main import MainWindow
from node import ImageNode, ToGrayscaleNode, BlurNode


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)

    # Synthetic BGR test image
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :80] = (255, 0, 0)  # left half blue

    window = MainWindow(img, "smoke")
    window.show()
    app.processEvents()

    scene = window.drop_widget.view._scene

    # The constructor added one ImageNode from the initial image.
    image_nodes = [it for it in scene.items() if isinstance(it, ImageNode)]
    assert len(image_nodes) == 1, f"expected 1 image node, got {len(image_nodes)}"
    src = image_nodes[0]
    assert src.get_output_image() is not None, "image node has no output"

    # Add a grayscale node and wire image -> grayscale.
    window.drop_widget.add_function_node("To Grayscale")
    gray = next(it for it in scene.items() if isinstance(it, ToGrayscaleNode))
    assert gray.can_accept_input(src)
    gray.add_input_connection(src)
    app.processEvents()
    out = gray.get_output_image()
    assert out is not None, "grayscale produced no output"
    assert out.ndim == 2, f"grayscale output should be single-channel, got shape {out.shape}"

    # Chain grayscale -> blur.
    window.drop_widget.add_function_node("Blur")
    blur = next(it for it in scene.items() if isinstance(it, BlurNode))
    blur.add_input_connection(gray)
    app.processEvents()
    assert blur.get_output_image() is not None, "blur produced no output"

    print("SMOKE OK: window built, image->grayscale->blur pipeline executed")
    print(f"  image node output shape: {src.get_output_image().shape}")
    print(f"  grayscale output shape:  {out.shape}")
    print(f"  blur output shape:       {blur.get_output_image().shape}")

    window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
