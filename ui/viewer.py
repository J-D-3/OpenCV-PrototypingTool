"""Standalone, pinned inspector window for one node (frontend).

Opened by double-clicking a node. Signal-driven (no polling): it subscribes to
the controller's nodeChanged and refreshes when its node recomputes. Shows the
node's preview image (with mouse-wheel zoom-to-cursor), size/type metadata, the
pixel under the cursor (x/y + per-channel values), an op summary, and — for
operation nodes — the node's parameters (editable, like the main pane).
"""
import numpy as np
from PyQt6 import QtWidgets

from ui.nodes import Node
from ui.image_utils import to_uint8
from ui.inspector_pane import ImagePanel, channel_names


class ImageViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, node: Node, parent=None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle(f"Inspector - {node._meta.get('name', 'Node')}")
        self.setMinimumSize(440, 380)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self._summary = QtWidgets.QLabel("")
        self._summary.setStyleSheet("font-weight: bold;")
        self._summary.setVisible(False)
        layout.addWidget(self._summary)

        self._meta = QtWidgets.QLabel("")
        self._meta.setStyleSheet("color: #666;")
        layout.addWidget(self._meta)

        self._image = ImagePanel()                 # wheel-zoom + hover, double-click = fit
        self._image.pixelHovered.connect(self._on_hover)
        layout.addWidget(self._image, 1)

        self._readout = QtWidgets.QLabel("—")
        self._readout.setStyleSheet("font-family: monospace;")
        layout.addWidget(self._readout)

        # (Parameters are edited only in the main window's panel.)

        self._disp = None          # uint8 display image for pixel sampling
        self._names = ["B", "G", "R"]
        self._controller = getattr(node, "controller", None)
        if self._controller is not None:
            self._controller.signals.nodeChanged.connect(self._on_node_changed)
        self.update_image()

    def _on_node_changed(self, qt_node) -> None:
        if qt_node is self.node:
            self.update_image()

    def _type_text(self, channels: int) -> str:
        space = (getattr(getattr(self.node, "gnode", None), "color_space", "") or "").lower()
        return {"bgr": "BGR", "hls": "HLS", "gray": "Gray", "binary": "Binary"}.get(
            space, "Gray" if channels == 1 else "BGR")

    def update_image(self):
        summary = self.node.get_summary()
        if summary:
            self._summary.setText("   ".join(f"{k}: {v}" for k, v in summary.items()))
            self._summary.setVisible(True)
        else:
            self._summary.setVisible(False)

        image = self.node.get_preview_image()
        if not isinstance(image, np.ndarray):
            self._disp = None
            self._image.set_image(None)
            self._meta.setText("")
            self._readout.setText("—")
            return

        self._disp = to_uint8(image)
        h, w = self._disp.shape[:2]
        channels = 1 if self._disp.ndim == 2 else self._disp.shape[2]
        self._names = channel_names(self.node, channels)
        self._meta.setText(f"{w}×{h}   {self._type_text(channels)}   {channels} ch")
        self._image.set_image(image)

    def _on_hover(self, x, y) -> None:
        if self._disp is None:
            return
        h, w = self._disp.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        px = self._disp[y, x]
        if self._disp.ndim == 3:
            vals = "  ".join(f"{nm}={int(v)}" for nm, v in zip(self._names, px))
        else:
            vals = f"{self._names[0]}={int(px)}"
        self._readout.setText(f"x={x} y={y}   {vals}")

    def closeEvent(self, event):
        if self._controller is not None:
            try:
                self._controller.signals.nodeChanged.disconnect(self._on_node_changed)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)
