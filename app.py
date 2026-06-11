"""Application entrypoint (argparse + QApplication)."""
import core._threadlimit  # noqa: F401 — MUST be first: pins OpenBLAS to 1 thread
                          # before numpy/cv2 load (prevents the batch-fan-out crash/hang)
import sys
import argparse
from pathlib import Path

import cv2
from PyQt6 import QtWidgets

from core import diag
from ui.main_window import MainWindow

def main() -> None:
    # Crash-hunting instrumentation: faulthandler dumps every thread's stack to
    # logs/faulthandler.log on a native crash; structural edits / evaluations are
    # logged to logs/diag.log. Set OCVPT_DIAG=1 for verbose per-node timing.
    diag.init()
    parser = argparse.ArgumentParser(description="OpenCV image viewer (PyQt6) with sidebar and drag-and-drop pane.")
    parser.add_argument("image_path", nargs="?", default=None, help="Optional path to an initial image")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    initial_image = None
    window_title = "OpenCV Image Viewer"

    if args.image_path is not None:
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"Error: File not found: '{image_path}'.", file=sys.stderr)
            sys.exit(1)
        initial_image = cv2.imread(str(image_path))
        if initial_image is None:
            print(f"Error: Could not load image from '{image_path}'. Check the path and file format.", file=sys.stderr)
            sys.exit(1)
        window_title = f"Image - {image_path.name}"

    window = MainWindow(initial_image, window_title)
    window.showMaximized()   # fill the screen by default (keeps the window frame)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
