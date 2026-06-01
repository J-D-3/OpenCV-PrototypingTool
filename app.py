"""Application entrypoint (argparse + QApplication)."""
import sys
import argparse
from pathlib import Path

import cv2
from PyQt6 import QtWidgets

from ui.main_window import MainWindow

def main() -> None:
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
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
