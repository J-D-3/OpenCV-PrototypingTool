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
from ui.widgets import LineSplitter


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
        self._margin = 5

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
        self.setMinimumHeight(90)
        self._curves = []    # list of (color, normalized np.ndarray[256])
        self._markers = []   # list of (color, lo, hi) — range handles to draw as lines
        self._pad_left = 0   # so the 0..255 span aligns with the sliders below
        self._pad_right = 0

    def set_padding(self, left: int, right: int) -> None:
        self._pad_left, self._pad_right = left, right
        self.update()

    def set_data(self, curves, markers):
        self._curves = curves
        self._markers = markers
        self.update()

    def _x(self, value, w):
        span = max(1, w - self._pad_left - self._pad_right)
        return self._pad_left + span * value / 255.0

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(250, 250, 250))
        w, h = self.width(), self.height()
        if w < 4:
            return

        # Range markers: a vertical line at each non-default handle (skip 0 / 255).
        for color, lo, hi in self._markers:
            p.setPen(QtGui.QPen(color, 1, QtCore.Qt.PenStyle.DashLine))
            if lo > 0:
                x = self._x(lo, w)
                p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
            if hi < 255:
                x = self._x(hi, w)
                p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))

        for color, norm in self._curves:
            p.setPen(QtGui.QPen(color, 1))
            path = QtGui.QPainterPath()
            for i in range(256):
                x = self._x(i, w)
                y = h - 1 - norm[i] * (h - 2)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.drawPath(path)


class HistogramPanel(QtWidgets.QWidget):
    rangesChanged = QtCore.pyqtSignal()

    _CB_W = 28          # checkbox+label column width
    _ROW_SPACING = 2    # gap between checkbox and slider
    _SLIDER_MARGIN = 5  # RangeSlider track inset (must match RangeSlider._margin)

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
        self._plot.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)
        # Align the plot's 0..255 span with the channel sliders below it:
        # left pad = checkbox column + row spacing + slider track margin.
        self._plot.set_padding(self._CB_W + self._ROW_SPACING + self._SLIDER_MARGIN,
                               self._SLIDER_MARGIN)
        self._layout.addWidget(self._plot, 1)   # plot takes the slack; rows stay compact
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
            row.setSpacing(self._ROW_SPACING)
            cb = QtWidgets.QCheckBox(names[i])
            cb.setChecked(True)
            cb.setFixedWidth(self._CB_W)
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
        self._frozen = False

    def set_patch(self, patch):
        self._patch = patch
        self.update()

    def set_frozen(self, frozen: bool):
        self._frozen = frozen
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
        # outline the centre cell — gold while live, crimson when frozen
        mid = n // 2
        outline = QtGui.QColor(220, 20, 60) if self._frozen else QtGui.QColor(255, 215, 0)
        p.setPen(QtGui.QPen(outline, 2))
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
        # One row: pixel coordinate + per-channel value (left), grid size (right).
        row = QtWidgets.QHBoxLayout()
        self._readout = QtWidgets.QLabel("—")
        self._readout.setStyleSheet("font-family: monospace;")
        row.addWidget(self._readout)
        row.addStretch()
        row.addWidget(QtWidgets.QLabel("Grid"))
        self._grid_combo = QtWidgets.QComboBox()
        for s in self.GRID_SIZES:
            self._grid_combo.addItem(f"{s}×{s}", s)
        self._grid_combo.setCurrentIndex(self.GRID_SIZES.index(27))  # default 27x27
        self._grid_combo.currentIndexChanged.connect(lambda _i: self._rebuild())
        row.addWidget(self._grid_combo)
        layout.addLayout(row)
        self._view = NeighborhoodView()
        self._view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Expanding)
        layout.addWidget(self._view, 1)

        self._base = None       # uint8 display image
        self._names = ["B", "G", "R"]
        self._center = None     # (x, y)
        self._frozen = False

    def set_frozen(self, frozen: bool) -> None:
        self._frozen = frozen
        self._view.set_frozen(frozen)
        self._rebuild()

    def set_base(self, disp, names) -> None:
        # Update the sampled image (e.g. the histogram-filtered preview) but keep
        # the current cursor/frozen position so re-rendering reflects the filter.
        self._base = disp
        self._names = names
        self._rebuild()

    def reset_center(self) -> None:
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
        h, w = self._base.shape[:2]
        if h == 0 or w == 0:
            return
        n = self.grid_size()
        half = n // 2
        # Clamp the centre to the current image — a frozen/hovered point from a
        # differently-sized batch frame can fall outside a smaller image.
        x = max(0, min(self._center[0], w - 1))
        y = max(0, min(self._center[1], h - 1))
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
        suffix = "   [frozen]" if self._frozen else ""
        self._readout.setText(f"x={x} y={y}   {vals}{suffix}")


# ---------------------------------------------------------------------------
# Image panel (hover/click aware)
# ---------------------------------------------------------------------------
class ImagePanel(QtWidgets.QWidget):
    pixelHovered = QtCore.pyqtSignal(int, int)
    pixelClicked = QtCore.pyqtSignal(int, int)        # left-click: freeze
    pixelReleased = QtCore.pyqtSignal(int, int)       # right-click: unfreeze

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
        xy = self._to_image_xy(e.position())
        if xy is None:
            return
        if e.button() == QtCore.Qt.MouseButton.LeftButton:
            self.pixelClicked.emit(*xy)
        elif e.button() == QtCore.Qt.MouseButton.RightButton:
            self.pixelReleased.emit(*xy)


# ---------------------------------------------------------------------------
# The pane
# ---------------------------------------------------------------------------
class InspectorPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(340)
        self.setMaximumWidth(580)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header: title (left) + batch frame nav "< i/N >" (right, only for batches).
        header = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("Inspector — (no selection)")
        self._title.setStyleSheet("font-weight: bold;")
        header.addWidget(self._title)
        header.addStretch()
        self._prev_btn = QtWidgets.QPushButton("<")
        self._next_btn = QtWidgets.QPushButton(">")
        self._prev_btn.setFixedWidth(26)
        self._next_btn.setFixedWidth(26)
        self._frame_label = QtWidgets.QLabel("")
        self._frame_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setMinimumWidth(44)
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn.clicked.connect(lambda: self._step(1))
        self._frame_nav = QtWidgets.QWidget()
        nav = QtWidgets.QHBoxLayout(self._frame_nav)
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(2)
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._frame_label)
        nav.addWidget(self._next_btn)
        self._frame_nav.setVisible(False)
        header.addWidget(self._frame_nav)
        layout.addLayout(header)

        # Image metadata: size + color type.
        self._meta = QtWidgets.QLabel("")
        self._meta.setStyleSheet("color: #666;")
        layout.addWidget(self._meta)

        # Three views, ~1/3 each, with a thin 1px handle (wide grab area) between.
        splitter = LineSplitter(QtCore.Qt.Orientation.Vertical)
        self._image = ImagePanel()
        self._neigh = NeighborhoodPanel()
        self._hist = HistogramPanel()
        for i, panel in enumerate((self._image, self._neigh, self._hist)):
            splitter.addWidget(panel)
            splitter.setStretchFactor(i, 1)
        splitter.setSizes([1000, 1000, 1000])  # start as equal thirds
        layout.addWidget(splitter, 1)

        self._node = None
        self._disp = None       # uint8 display image (unfiltered)
        self._channels = 0
        self._names = []
        self._frozen = False

        self._image.setToolTip("Left-click: freeze the pixel readout · Right-click: release")
        self._image.pixelHovered.connect(self._on_hover)
        self._image.pixelClicked.connect(self._on_click)
        self._image.pixelReleased.connect(self._on_release)
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
    def _batch_len(self) -> int:
        value = self._node._batch_value() if self._node is not None else None
        return len(value) if isinstance(value, Batch) else 1

    def _step(self, delta: int) -> None:
        if self._node is None or self._node.controller is None:
            return
        n = self._batch_len()
        idx = max(0, min(self._node.controller.preview_index + delta, n - 1))
        self._node.controller.set_preview_index(idx)  # emits -> pane refreshes

    def _update_frame_controls(self) -> None:
        n = self._batch_len()
        controller = getattr(self._node, "controller", None)
        if n > 1 and controller is not None:
            idx = min(controller.preview_index, n - 1)
            self._frame_label.setText(f"{idx + 1}/{n}")
            self._prev_btn.setEnabled(idx > 0)
            self._next_btn.setEnabled(idx < n - 1)
            self._frame_nav.setVisible(True)
        else:
            self._frame_nav.setVisible(False)

    @staticmethod
    def _type_text(node, channels) -> str:
        space = (getattr(getattr(node, "gnode", None), "color_space", "") or "").lower()
        label = {"bgr": "BGR", "hls": "HLS", "gray": "Gray", "binary": "Binary"}.get(space)
        if label is None:
            label = "Gray" if channels == 1 else "BGR"
        return label

    def _recompute(self, reset: bool) -> None:
        self._update_frame_controls()
        raw = self._node.get_preview_image() if self._node is not None else None
        if not isinstance(raw, np.ndarray):
            self._disp = None
            self._channels = 0
            self._names = []
            self._meta.setText("")
            self._image.set_image(None)
            self._neigh.reset_center()
            self._neigh.set_base(None, [])
            self._hist.clear()
            return

        disp = to_uint8(raw)
        self._disp = disp
        channels = 1 if disp.ndim == 2 else disp.shape[2]
        self._names = channel_names(self._node, channels)
        h, w = disp.shape[:2]
        self._meta.setText(f"{w}×{h}   {self._type_text(self._node, channels)}   {channels} ch")

        if reset or channels != self._channels:
            self._hist.configure(channels, self._names)
            self._channels = channels
        if reset:
            self._neigh.reset_center()
        self._hist.set_hists(self._compute_hists(disp, channels))
        self._apply_filter()   # updates both the image and the neighbourhood

    @staticmethod
    def _compute_hists(disp, channels):
        if channels == 1:
            return [cv2.calcHist([disp], [0], None, [256], [0, 256]).flatten()]
        return [cv2.calcHist([disp], [c], None, [256], [0, 256]).flatten()
                for c in range(channels)]

    def _filtered_image(self):
        """The display image with the histogram ranges masked out (pixels outside
        any active range set to black). Returns None if no node/image."""
        if self._disp is None:
            return None
        disp = self._disp
        mask = np.ones(disp.shape[:2], dtype=bool)
        for i, (enabled, lo, hi) in enumerate(self._hist.ranges()):
            if not enabled or (lo <= 0 and hi >= 255):
                continue
            ch = disp if disp.ndim == 2 else disp[:, :, i]
            mask &= (ch >= lo) & (ch <= hi)
        if mask.all():
            return disp
        out = disp.copy()
        out[~mask] = 0
        return out

    def _apply_filter(self) -> None:
        shown = self._filtered_image()
        if shown is None:
            return
        # The image and the pixel-neighbourhood both reflect the filtered view.
        self._image.set_image(shown)
        self._neigh.set_base(shown, self._names)

    def _on_hover(self, x, y) -> None:
        if not self._frozen:
            self._neigh.set_center(x, y)

    def _on_click(self, x, y) -> None:
        # Left-click freezes the neighbourhood on this pixel (re-clicking moves it).
        self._frozen = True
        self._neigh.set_frozen(True)
        self._neigh.set_center(x, y)

    def _on_release(self, x, y) -> None:
        # Right-click releases the freeze and resumes live cursor tracking.
        self._frozen = False
        self._neigh.set_frozen(False)
        self._neigh.set_center(x, y)
