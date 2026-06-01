"""Live inspector pane (frontend).

A docked pane that always reflects the *currently selected* node (unlike the
free-floating ImageViewerWindow, which is pinned to one node). It stacks three
sub-panels vertically:

  1. Image    — the node's output / render_preview, optionally masked by the
                histogram range filter below.
  2. Pixels   — the neighbourhood (3..81 grid) around the hovered/frozen pixel,
                with its position and per-channel value.
  3. Histogram— per-channel curves with toggles and draggable min/max ranges;
                narrowing a range masks the image above to in-range pixels.

Everything operates on a uint8 "display" copy of the image (floats normalized),
so what you see, hover, and histogram all agree.
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from core.batch import Batch
from ui.image_utils import cv_to_qimage, to_uint8


# ---------------------------------------------------------------------------
# channel naming / colors
# ---------------------------------------------------------------------------
def channel_names(node, channels: int):
    if channels == 1:
        return ["Gray"]
    # Prefer the engine-tracked color space, fall back to the op's output label.
    space = (getattr(getattr(node, "gnode", None), "color_space", "") or "").lower()
    out = ""
    op = getattr(node, "op", None)
    if op is not None:
        out = (op.out_label or "").upper()
    if space == "hls" or "HLS" in out or "HSL" in out:
        return ["H", "L", "S"]
    if space == "hsv" or "HSV" in out:
        return ["H", "S", "V"]
    if "LAB" in out:
        return ["L", "a", "b"]
    return ["B", "G", "R"]


# Draw color per channel *name* (so HLS/HSV curves aren't mislabelled blue/green/red).
_CHANNEL_COLORS = {
    "B": QtGui.QColor(40, 90, 220),     # blue
    "G": QtGui.QColor(30, 160, 60),     # green
    "R": QtGui.QColor(210, 50, 50),     # red
    "Gray": QtGui.QColor(60, 60, 60),   # dark gray (contrast on light bg)
    "H": QtGui.QColor(200, 0, 200),     # magenta
    "S": QtGui.QColor(0, 150, 150),     # teal
    "V": QtGui.QColor(90, 90, 90),      # dark gray
    "L": QtGui.QColor(120, 120, 120),   # mid gray
    "a": QtGui.QColor(170, 100, 40),    # olive/brown
    "b": QtGui.QColor(40, 100, 170),    # steel blue
}


def _channel_color(name: str) -> QtGui.QColor:
    return _CHANNEL_COLORS.get(name, QtGui.QColor(40, 90, 220))


# ---------------------------------------------------------------------------
# RangeSlider — a two-handle 0..255 selector
# ---------------------------------------------------------------------------
class RangeSlider(QtWidgets.QWidget):
    rangeChanged = QtCore.pyqtSignal()

    def __init__(self, color: QtGui.QColor, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setMinimumWidth(140)
        self._lo, self._hi = 0, 255
        self._color = color
        self._drag = None  # 'lo' | 'hi' | None
        self._margin = 6

    def values(self):
        return self._lo, self._hi

    def is_full(self) -> bool:
        return self._lo <= 0 and self._hi >= 255

    def reset(self):
        self._lo, self._hi = 0, 255
        self.update()

    def _track_rect(self):
        return QtCore.QRect(self._margin, self.height() // 2 - 2,
                            self.width() - 2 * self._margin, 4)

    def _val_to_x(self, v):
        t = self._track_rect()
        return t.left() + int(t.width() * v / 255)

    def _x_to_val(self, x):
        t = self._track_rect()
        if t.width() <= 0:
            return 0
        return int(round(255 * (x - t.left()) / t.width()))

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        t = self._track_rect()
        p.fillRect(t, QtGui.QColor(210, 210, 210))
        x_lo, x_hi = self._val_to_x(self._lo), self._val_to_x(self._hi)
        sel = QtGui.QColor(self._color)
        sel.setAlpha(140)
        p.fillRect(QtCore.QRect(x_lo, t.top(), max(1, x_hi - x_lo), t.height()), sel)
        p.setBrush(self._color)
        p.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40)))
        for x in (x_lo, x_hi):
            p.drawRoundedRect(QtCore.QRectF(x - 4, self.height() / 2 - 8, 8, 16), 2, 2)

    def mousePressEvent(self, e):
        x = e.position().x()
        self._drag = 'lo' if abs(x - self._val_to_x(self._lo)) <= abs(x - self._val_to_x(self._hi)) else 'hi'
        self._apply(x)

    def mouseMoveEvent(self, e):
        if self._drag:
            self._apply(e.position().x())

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def _apply(self, x):
        v = max(0, min(255, self._x_to_val(x)))
        if self._drag == 'lo':
            self._lo = min(v, self._hi)
        else:
            self._hi = max(v, self._lo)
        self.update()
        self.rangeChanged.emit()


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
class HistogramPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._curves = []    # list of (color, normalized np.ndarray[256])
        self._markers = []   # list of (color, lo, hi) — range handles to draw as lines

    def set_data(self, curves, markers):
        self._curves = curves
        self._markers = markers
        self.update()

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(250, 250, 250))
        w, h = self.width(), self.height()
        if w < 4:
            return

        # Range markers: a vertical line at each non-default handle (skip 0 / 255).
        for color, lo, hi in self._markers:
            pen = QtGui.QPen(color, 1, QtCore.Qt.PenStyle.DashLine)
            p.setPen(pen)
            if lo > 0:
                x = w * lo / 255
                p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
            if hi < 255:
                x = w * hi / 255
                p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))

        for color, norm in self._curves:
            p.setPen(QtGui.QPen(color, 1))
            path = QtGui.QPainterPath()
            for i in range(256):
                x = w * i / 255
                y = h - 1 - norm[i] * (h - 2)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.drawPath(path)


class HistogramPanel(QtWidgets.QWidget):
    rangesChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(2)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Histogram"))
        header.addStretch()
        self._log_cb = QtWidgets.QCheckBox("Log scale")
        self._log_cb.toggled.connect(self._refresh_plot)
        header.addWidget(self._log_cb)
        self._layout.addLayout(header)
        self._plot = HistogramPlot()
        self._layout.addWidget(self._plot)
        self._rows_box = QtWidgets.QWidget()
        self._rows = QtWidgets.QVBoxLayout(self._rows_box)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(1)
        self._layout.addWidget(self._rows_box)
        self._channels = []   # list of dicts: {name, color, checkbox, slider}
        self._hists = []      # raw per-channel histograms (256,)

    def configure(self, n_channels: int, names) -> None:
        """(Re)build one toggle+range row per channel (resets ranges).

        Each row is wrapped in a QWidget so deleting it removes its child
        widgets too — otherwise old channel labels linger and overlap.
        """
        while self._rows.count():
            child = self._rows.takeAt(0)
            w = child.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._channels = []
        for i in range(n_channels):
            color = _channel_color(names[i])
            row_w = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            cb = QtWidgets.QCheckBox(names[i])
            cb.setChecked(True)
            cb.setFixedWidth(48)
            cb.setStyleSheet(f"color: rgb({color.red()},{color.green()},{color.blue()});")
            slider = RangeSlider(color)
            cb.toggled.connect(self._on_changed)
            slider.rangeChanged.connect(self._on_changed)
            row.addWidget(cb)
            row.addWidget(slider)
            self._rows.addWidget(row_w)
            self._channels.append({"name": names[i], "color": color, "cb": cb, "slider": slider})

    def set_hists(self, hists) -> None:
        self._hists = hists
        self._refresh_plot()

    def ranges(self):
        """Per channel: (enabled, lo, hi)."""
        out = []
        for ch in self._channels:
            lo, hi = ch["slider"].values()
            out.append((ch["cb"].isChecked(), lo, hi))
        return out

    def clear(self):
        self.configure(0, [])
        self._hists = []
        self._plot.set_data([], [])

    def _on_changed(self):
        self._refresh_plot()
        self.rangesChanged.emit()

    def _refresh_plot(self):
        log = self._log_cb.isChecked()
        curves = []
        markers = []
        for i, ch in enumerate(self._channels):
            if i < len(self._hists) and ch["cb"].isChecked():
                h = self._hists[i].astype(np.float32)
                if log:
                    h = np.log1p(h)
                peak = float(h.max())
                norm = h / peak if peak > 0 else h
                curves.append((ch["color"], norm))
                lo, hi = ch["slider"].values()
                markers.append((ch["color"], lo, hi))
        self._plot.set_data(curves, markers)


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------
class NeighborhoodView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._patch = None  # (N, N, 3) uint8 BGR, or None

    def set_patch(self, patch):
        self._patch = patch
        self.update()

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        # Light-gray backdrop that shows through the gaps as cell separators.
        p.fillRect(self.rect(), QtGui.QColor(205, 205, 205))
        if self._patch is None:
            return
        n = self._patch.shape[0]
        size = min(self.width(), self.height())
        cell = size / n
        ox = (self.width() - cell * n) / 2
        oy = (self.height() - cell * n) / 2
        gap = max(1.0, cell * 0.12)
        for r in range(n):
            for c in range(n):
                b, g, rd = (int(v) for v in self._patch[r, c])
                p.fillRect(QtCore.QRectF(ox + c * cell + gap / 2, oy + r * cell + gap / 2,
                                         cell - gap, cell - gap),
                           QtGui.QColor(rd, g, b))
        # outline the centre cell
        mid = n // 2
        p.setPen(QtGui.QPen(QtGui.QColor(255, 215, 0), 2))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(QtCore.QRectF(ox + mid * cell + gap / 2, oy + mid * cell + gap / 2,
                                 cell - gap, cell - gap))


class NeighborhoodPanel(QtWidgets.QWidget):
    GRID_SIZES = [3, 9, 27, 81]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Pixels"))
        self._grid_combo = QtWidgets.QComboBox()
        for s in self.GRID_SIZES:
            self._grid_combo.addItem(f"{s}×{s}", s)
        self._grid_combo.setCurrentIndex(1)  # 9x9
        self._grid_combo.currentIndexChanged.connect(lambda _i: self._rebuild())
        top.addWidget(self._grid_combo)
        top.addStretch()
        layout.addLayout(top)
        self._view = NeighborhoodView()
        layout.addWidget(self._view)
        self._readout = QtWidgets.QLabel("—")
        self._readout.setStyleSheet("font-family: monospace;")
        layout.addWidget(self._readout)

        self._base = None       # uint8 display image
        self._names = ["B", "G", "R"]
        self._center = None     # (x, y)

    def set_base(self, disp, names) -> None:
        self._base = disp
        self._names = names
        self._center = None
        self._readout.setText("—")
        self._view.set_patch(None)

    def set_center(self, x, y) -> None:
        self._center = (x, y)
        self._rebuild()

    def grid_size(self) -> int:
        return self._grid_combo.currentData()

    def _rebuild(self):
        if self._base is None or self._center is None:
            return
        n = self.grid_size()
        half = n // 2
        x, y = self._center
        h, w = self._base.shape[:2]
        patch = np.full((n, n, 3), 200, np.uint8)  # gray for out-of-bounds
        for r in range(n):
            for c in range(n):
                sy, sx = y - half + r, x - half + c
                if 0 <= sy < h and 0 <= sx < w:
                    px = self._base[sy, sx]
                    patch[r, c] = px if self._base.ndim == 3 else (int(px),) * 3
        self._view.set_patch(patch)

        px = self._base[y, x]
        if self._base.ndim == 3:
            vals = "  ".join(f"{nm}={int(v)}" for nm, v in zip(self._names, px))
        else:
            vals = f"{self._names[0]}={int(px)}"
        self._readout.setText(f"x={x} y={y}   {vals}")


# ---------------------------------------------------------------------------
# Image panel (hover/click aware)
# ---------------------------------------------------------------------------
class ImagePanel(QtWidgets.QWidget):
    pixelHovered = QtCore.pyqtSignal(int, int)
    pixelClicked = QtCore.pyqtSignal(int, int)

    _MIN_SCALE = 0.02
    _MAX_SCALE = 64.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self._pixmap = QtGui.QPixmap()
        self._img_w = 0
        self._img_h = 0
        self._scale = 1.0                  # widget px per image px
        self._origin = QtCore.QPointF(0, 0)  # widget pos of image (0,0)
        self._fitted = False               # False -> recompute fit on next paint

    def set_image(self, image) -> None:
        prev = (self._img_w, self._img_h)
        if image is None:
            self._pixmap = QtGui.QPixmap()
            self._img_w = self._img_h = 0
        else:
            self._img_h, self._img_w = image.shape[:2]
            self._pixmap = QtGui.QPixmap.fromImage(cv_to_qimage(image))
        # Refit only when the image size changes (so a histogram-filter update,
        # which keeps the same size, preserves the user's current zoom).
        if (self._img_w, self._img_h) != prev:
            self._fitted = False
        self.update()

    def _fit(self) -> None:
        if self._pixmap.isNull() or self._img_w == 0 or self.width() <= 0:
            return
        self._scale = min(self.width() / self._img_w, self.height() / self._img_h)
        self._origin = QtCore.QPointF((self.width() - self._img_w * self._scale) / 2,
                                      (self.height() - self._img_h * self._scale) / 2)
        self._fitted = True

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(30, 30, 30))
        if self._pixmap.isNull():
            return
        if not self._fitted:
            self._fit()
        target = QtCore.QRectF(self._origin.x(), self._origin.y(),
                               self._img_w * self._scale, self._img_h * self._scale)
        p.drawPixmap(target, self._pixmap, QtCore.QRectF(self._pixmap.rect()))

    def resizeEvent(self, e):
        self._fitted = False  # refit to the new size
        super().resizeEvent(e)

    def wheelEvent(self, e):
        if self._pixmap.isNull():
            return
        self._zoom_at(e.position(), 1.2 ** (e.angleDelta().y() / 120.0))

    def _zoom_at(self, pos, factor) -> None:
        """Scale by ``factor`` keeping the image point under ``pos`` fixed."""
        if self._pixmap.isNull():
            return
        if not self._fitted:
            self._fit()
        ix = (pos.x() - self._origin.x()) / self._scale  # image point under cursor
        iy = (pos.y() - self._origin.y()) / self._scale
        self._scale = max(self._MIN_SCALE, min(self._MAX_SCALE, self._scale * factor))
        self._origin = QtCore.QPointF(pos.x() - ix * self._scale, pos.y() - iy * self._scale)
        self._fitted = True  # custom zoom; don't auto-refit until size changes
        self.update()

    def mouseDoubleClickEvent(self, _e):
        self._fitted = False  # reset to fit
        self.update()

    def _to_image_xy(self, pos):
        if self._pixmap.isNull() or self._scale <= 0:
            return None
        x = int((pos.x() - self._origin.x()) / self._scale)
        y = int((pos.y() - self._origin.y()) / self._scale)
        if 0 <= x < self._img_w and 0 <= y < self._img_h:
            return x, y
        return None

    def mouseMoveEvent(self, e):
        xy = self._to_image_xy(e.position())
        if xy is not None:
            self.pixelHovered.emit(*xy)

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            xy = self._to_image_xy(e.position())
            if xy is not None:
                self.pixelClicked.emit(*xy)


# ---------------------------------------------------------------------------
# The pane
# ---------------------------------------------------------------------------
class InspectorPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setMaximumWidth(520)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._title = QtWidgets.QLabel("Inspector — (no selection)")
        self._title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._title)

        # Batch frame selector (visible only when the selected node holds >1 image).
        frame_row = QtWidgets.QHBoxLayout()
        self._frame_label = QtWidgets.QLabel("")
        self._frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._frame_slider.valueChanged.connect(self._on_frame)
        frame_row.addWidget(self._frame_label)
        frame_row.addWidget(self._frame_slider)
        self._frame_widget = QtWidgets.QWidget()
        self._frame_widget.setLayout(frame_row)
        self._frame_widget.setVisible(False)
        layout.addWidget(self._frame_widget)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._image = ImagePanel()
        self._neigh = NeighborhoodPanel()
        self._hist = HistogramPanel()
        splitter.addWidget(self._image)
        splitter.addWidget(self._neigh)
        splitter.addWidget(self._hist)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)
        layout.addWidget(splitter)

        self._node = None
        self._disp = None       # uint8 display image (unfiltered)
        self._channels = 0
        self._frozen = False

        self._image.pixelHovered.connect(self._on_hover)
        self._image.pixelClicked.connect(self._on_click)
        self._neigh._grid_combo.currentIndexChanged.connect(lambda _i: None)
        self._hist.rangesChanged.connect(self._apply_filter)

    # --- public API --------------------------------------------------------
    def set_node(self, node) -> None:
        self._node = node
        self._frozen = False
        name = node._meta.get("name", "Node") if node is not None else "(no selection)"
        self._title.setText(f"Inspector — {name}")
        self._recompute(reset=True)

    def refresh(self) -> None:
        """Re-read the node's result (e.g. after a parameter change)."""
        if self._node is not None:
            self._recompute(reset=False)

    # --- internals ---------------------------------------------------------
    def _on_frame(self, value: int) -> None:
        if self._node is None or self._node.controller is None:
            return
        self._node.controller.set_preview_index(value)  # re-render every node thumbnail
        self._recompute(reset=False)                    # and the pane's own element

    def _update_frame_controls(self) -> None:
        full = self._node.gnode.output if (self._node is not None and getattr(self._node, "gnode", None)) else None
        n = len(full) if isinstance(full, Batch) else 1
        controller = getattr(self._node, "controller", None)
        if n > 1 and controller is not None:
            idx = min(controller.preview_index, n - 1)
            self._frame_slider.blockSignals(True)
            self._frame_slider.setRange(0, n - 1)
            self._frame_slider.setValue(idx)
            self._frame_slider.blockSignals(False)
            self._frame_label.setText(f"Image {idx + 1}/{n}")
            self._frame_widget.setVisible(True)
        else:
            self._frame_widget.setVisible(False)

    def _recompute(self, reset: bool) -> None:
        self._update_frame_controls()
        raw = self._node.get_preview_image() if self._node is not None else None
        if not isinstance(raw, np.ndarray):
            self._disp = None
            self._channels = 0
            self._image.set_image(None)
            self._neigh.set_base(None, [])
            self._hist.clear()
            return

        disp = to_uint8(raw)
        self._disp = disp
        channels = 1 if disp.ndim == 2 else disp.shape[2]
        names = channel_names(self._node, channels)

        if reset or channels != self._channels:
            self._hist.configure(channels, names)
            self._channels = channels
        self._hist.set_hists(self._compute_hists(disp, channels))
        self._neigh.set_base(disp, names)
        self._apply_filter()

    @staticmethod
    def _compute_hists(disp, channels):
        if channels == 1:
            return [cv2.calcHist([disp], [0], None, [256], [0, 256]).flatten()]
        return [cv2.calcHist([disp], [c], None, [256], [0, 256]).flatten()
                for c in range(channels)]

    def _apply_filter(self) -> None:
        if self._disp is None:
            return
        disp = self._disp
        mask = np.ones(disp.shape[:2], dtype=bool)
        for i, (enabled, lo, hi) in enumerate(self._hist.ranges()):
            if not enabled or (lo <= 0 and hi >= 255):
                continue
            ch = disp if disp.ndim == 2 else disp[:, :, i]
            mask &= (ch >= lo) & (ch <= hi)
        if mask.all():
            self._image.set_image(disp)
        else:
            filtered = disp.copy()
            filtered[~mask] = 0
            self._image.set_image(filtered)

    def _on_hover(self, x, y) -> None:
        if not self._frozen:
            self._neigh.set_center(x, y)

    def _on_click(self, x, y) -> None:
        # Left-click freezes the neighbourhood on this pixel; click again to unfreeze.
        if self._frozen and self._neigh._center == (x, y):
            self._frozen = False
        else:
            self._frozen = True
            self._neigh.set_center(x, y)
