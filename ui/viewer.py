"""Standalone, pinned inspector window for one node (frontend).

Opened by double-clicking a node. Signal-driven (no polling): it subscribes to
the controller's nodeChanged and refreshes when its node recomputes. Shows the
node's preview image (with mouse-wheel zoom-to-cursor), size/type metadata, the
pixel under the cursor (x/y + per-channel values), an op summary, and — for
operation nodes — the node's parameters (editable, like the main pane).
"""
import numpy as np
from PyQt6 import QtWidgets

from core.batch import Batch
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

        # Batch-frame control. By default the window FOLLOWS the global preview
        # index — scrub a batch on the canvas and this window updates with it. Tick
        # the checkbox to PIN it to the current frame, so it stays put while you
        # scrub others (open several windows pinned to different frames to compare).
        self._pinned_index = None                  # None = follow the global index
        self._frame_bar = QtWidgets.QWidget()
        fb = QtWidgets.QHBoxLayout(self._frame_bar)
        fb.setContentsMargins(0, 0, 0, 0)
        self._frame_label = QtWidgets.QLabel("")
        self._frame_label.setStyleSheet("color: #666;")
        self._pin_check = QtWidgets.QCheckBox("Pin to frame")
        self._pin_check.setToolTip(
            "Off: follow the canvas frame (updates as you scrub the batch).\n"
            "On: lock this window to the current frame.")
        self._pin_check.toggled.connect(self._on_pin_toggled)
        fb.addWidget(self._frame_label)
        fb.addStretch(1)
        fb.addWidget(self._pin_check)
        layout.addWidget(self._frame_bar)

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
            # Follow batch scrubbing on the canvas (unless this window is pinned).
            self._controller.signals.previewIndexChanged.connect(self._on_preview_index_changed)
        self.update_image()

    def _on_node_changed(self, qt_node) -> None:
        if qt_node is self.node:
            self.update_image()

    def _on_preview_index_changed(self, _index) -> None:
        # The canvas frame changed: re-render to follow it — unless we're pinned.
        if self._pinned_index is None:
            self.update_image()

    def _on_pin_toggled(self, checked: bool) -> None:
        if checked:
            value = self.node._batch_value()       # pin to whatever frame is shown now
            self._pinned_index = self.node._cur_index(value, None)
        else:
            self._pinned_index = None              # resume following the global index
        self.update_image()

    def _refresh_frame_bar(self, index) -> None:
        """Show 'Frame i/N' + the pin state; hide entirely when the node isn't a
        multi-element batch (pinning a single image is meaningless)."""
        value = self.node._batch_value()
        batched = isinstance(value, Batch) and len(value.items) > 1
        self._frame_bar.setVisible(batched)
        if not batched:
            return
        idx = self.node._cur_index(value, index)
        state = "pinned" if self._pinned_index is not None else "following"
        self._frame_label.setText(f"Frame {idx + 1}/{len(value.items)}  ({state})")
        self._pin_check.blockSignals(True)         # reflect state without re-firing
        self._pin_check.setChecked(self._pinned_index is not None)
        self._pin_check.setText(f"Pin to frame {idx + 1}")
        self._pin_check.blockSignals(False)

    def _type_text(self, channels: int) -> str:
        space = (getattr(getattr(self.node, "gnode", None), "color_space", "") or "").lower()
        return {"bgr": "BGR", "hls": "HLS", "gray": "Gray", "binary": "Binary"}.get(
            space, "Gray" if channels == 1 else "BGR")

    def update_image(self):
        index = self._pinned_index            # None = follow the global preview index
        self._refresh_frame_bar(index)

        summary = self.node.get_summary(index)
        if summary:
            self._summary.setText("   ".join(f"{k}: {v}" for k, v in summary.items()))
            self._summary.setVisible(True)
        else:
            self._summary.setVisible(False)

        image = self.node.get_preview_image(index)
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
            for sig, slot in ((self._controller.signals.nodeChanged, self._on_node_changed),
                              (self._controller.signals.previewIndexChanged,
                               self._on_preview_index_changed)):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        super().closeEvent(event)
