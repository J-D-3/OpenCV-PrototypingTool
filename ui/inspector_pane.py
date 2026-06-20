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
from ui.image_utils import cv_to_qimage, to_uint8, downscale_max
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


# Per-channel raw value range. OpenCV stores Hue as 0..179 (2 deg / unit); every
# other channel is 0..255.
def _ch_vmax(name: str) -> int:
    return 179 if name == "H" else 255


def _ch_value_label(name: str, raw: int) -> str:
    """Human value for a raw channel reading: Hue in degrees (0..358); Lightness and
    Saturation as a percentage (the HLS L/S byte is 0..255 = 0..100%); raw byte for
    intensity channels (B/G/R/Gray)."""
    if name == "H":
        return f"{int(round(raw * 2))}°"
    if name in ("L", "S"):
        return f"{int(round(raw / 255 * 100))}%"
    return str(int(raw))


# ---------------------------------------------------------------------------
# RangeSlider — a two-handle 0..255 selector
# ---------------------------------------------------------------------------
class RangeSlider(QtWidgets.QWidget):
    rangeChanged = QtCore.pyqtSignal()

    def __init__(self, color: QtGui.QColor, parent=None, vmax: int = 255):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setMinimumWidth(140)
        self._vmax = vmax
        self._lo, self._hi = 0, vmax
        self._color = color
        self._drag = None  # 'lo' | 'hi' | None
        self._margin = 5

    def set_vmax(self, vmax: int) -> None:
        self._vmax = max(1, int(vmax))
        self._lo, self._hi = 0, self._vmax
        self.update()

    def values(self):
        return self._lo, self._hi

    def is_full(self) -> bool:
        return self._lo <= 0 and self._hi >= self._vmax

    def reset(self):
        self._lo, self._hi = 0, self._vmax
        self.update()

    def set_values(self, lo, hi):
        """Restore a previously-set range (clamped to the slider's bounds)."""
        self._lo = max(0, min(self._vmax, int(lo)))
        self._hi = max(self._lo, min(self._vmax, int(hi)))
        self.update()

    def _track_rect(self):
        return QtCore.QRect(self._margin, self.height() // 2 - 2,
                            self.width() - 2 * self._margin, 4)

    def _val_to_x(self, v):
        t = self._track_rect()
        return t.left() + int(t.width() * v / self._vmax)

    def _x_to_val(self, x):
        t = self._track_rect()
        if t.width() <= 0:
            return 0
        return int(round(self._vmax * (x - t.left()) / t.width()))

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
        v = max(0, min(self._vmax, self._x_to_val(x)))
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

    def _x(self, value, w, vmax):
        span = max(1, w - self._pad_left - self._pad_right)
        return self._pad_left + span * value / float(vmax)

    def _draw_marker(self, p, color, x, h, raw, name):
        """A dashed line + an 'x = <value>' tag (Hue in degrees) at the handle."""
        p.setPen(QtGui.QPen(color, 1, QtCore.Qt.PenStyle.DashLine))
        p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
        text = f"{name}={_ch_value_label(name, raw)}"   # e.g. "H=170°"
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        tx = x + 2 if x + 2 + tw < self.width() else x - 2 - tw   # keep on-screen
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):          # halo for legibility
            p.drawText(QtCore.QPointF(tx + dx, fm.ascent() + 1 + dy), text)
        p.setPen(QtGui.QPen(color))
        p.drawText(QtCore.QPointF(tx, fm.ascent() + 1), text)

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(250, 250, 250))
        w, h = self.width(), self.height()
        if w < 4:
            return

        # Range markers: a vertical line + value tag at each non-default handle.
        for color, lo, hi, vmax, name in self._markers:
            if lo > 0:
                self._draw_marker(p, color, self._x(lo, w, vmax), h, lo, name)
            if hi < vmax:
                self._draw_marker(p, color, self._x(hi, w, vmax), h, hi, name)

        for color, norm, vmax in self._curves:
            p.setPen(QtGui.QPen(color, 1))
            path = QtGui.QPainterPath()
            n = len(norm)
            for i in range(n):
                x = self._x(i, w, vmax)
                y = h - 1 - norm[i] * (h - 2)
                (path.moveTo if i == 0 else path.lineTo)(x, y)
            p.drawPath(path)


class HistogramPanel(QtWidgets.QWidget):
    rangesChanged = QtCore.pyqtSignal()
    viewChanged = QtCore.pyqtSignal()      # BGR <-> HSL toggle

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
        # BGR <-> HSL view (only meaningful for a 3-channel image).
        self._view_combo = QtWidgets.QComboBox()
        self._view_combo.addItems(["BGR", "HSL"])
        self._view_combo.currentIndexChanged.connect(lambda _i: self.viewChanged.emit())
        header.addWidget(self._view_combo)
        header.addWidget(self._vsep())
        self._log_cb = QtWidgets.QCheckBox("Log scale")
        self._log_cb.toggled.connect(self._refresh_plot)
        header.addWidget(self._log_cb)
        header.addWidget(self._vsep())
        # Gaussian smoothing of the plotted curves (display only — does not affect
        # the range filter). 0 = off; higher = smoother (sigma in histogram bins).
        # Slider + editable value field, mirroring the parameter-panel sliders.
        header.addWidget(QtWidgets.QLabel("Smooth"))
        self._smooth = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._smooth.setRange(0, 15)
        self._smooth.setValue(0)
        self._smooth.setFixedWidth(70)
        smooth_tip = "Gaussian smoothing of the histogram curve (display only); 0 = off"
        self._smooth.setToolTip(smooth_tip)
        self._smooth.valueChanged.connect(self._on_smooth_slider)
        header.addWidget(self._smooth)
        self._smooth_field = QtWidgets.QLineEdit("0")
        self._smooth_field.setFixedWidth(30)
        self._smooth_field.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                        | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._smooth_field.setToolTip(smooth_tip)
        self._smooth_field.editingFinished.connect(self._on_smooth_field_edited)
        header.addWidget(self._smooth_field)
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
        # Per-channel (checked, (lo, hi)) by channel name, kept across rebuilds /
        # node switches so the user's view (isolated channels, narrowed ranges)
        # sticks — even through the intermediate clear() on deselection.
        self._saved_state = {}

    def configure(self, n_channels: int, names) -> None:
        """(Re)build one toggle+range row per channel.

        The per-channel toggle and range are **preserved across a rebuild for any
        channel whose name survives** — so switching to another node with the same
        channels keeps the user's view (isolated channels, narrowed ranges) instead
        of resetting it. Channels that don't carry over (e.g. Gray -> B/G/R, or a
        BGR<->HSL switch) fall back to checked + full range.

        Each row is wrapped in a QWidget so deleting it removes its child
        widgets too — otherwise old channel labels linger and overlap.
        """
        # Remember the live state of the channels we're about to tear down (so a
        # configure(0, []) on deselection doesn't drop it), then restore by name.
        for ch in self._channels:
            self._saved_state[ch["name"]] = (ch["cb"].isChecked(), ch["slider"].values())
        while self._rows.count():
            child = self._rows.takeAt(0)
            w = child.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._channels = []
        self._view_combo.setVisible(n_channels == 3)   # only colour images toggle
        for i in range(n_channels):
            color = _channel_color(names[i])
            vmax = _ch_vmax(names[i])
            checked, rng = self._saved_state.get(names[i], (True, None))
            row_w = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(self._ROW_SPACING)
            # The checkbox label IS the channel indicator (B/G/R/H/S/L), in colour.
            cb = QtWidgets.QCheckBox(names[i])
            cb.setChecked(checked)
            cb.setFixedWidth(self._CB_W)
            cb.setStyleSheet(f"color: rgb({color.red()},{color.green()},{color.blue()}); font-weight: bold;")
            cb.setToolTip(f"{names[i]} channel (0..{vmax})")
            slider = RangeSlider(color, vmax=vmax)
            if rng is not None:
                slider.set_values(rng[0], rng[1])
            cb.toggled.connect(self._on_changed)
            slider.rangeChanged.connect(self._on_changed)
            row.addWidget(cb)
            row.addWidget(slider)
            self._rows.addWidget(row_w)
            self._channels.append({"name": names[i], "color": color, "vmax": vmax,
                                   "cb": cb, "slider": slider})

    def color_view(self) -> str:
        return "hsl" if self._view_combo.currentText() == "HSL" else "bgr"

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

    @staticmethod
    def _vsep() -> QtWidgets.QFrame:
        """A minimal vertical separator between header control groups."""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    def _on_smooth_slider(self, value: int) -> None:
        self._smooth_field.setText(str(int(value)))
        self._refresh_plot()

    def _on_smooth_field_edited(self) -> None:
        """Type an exact smoothing value; clamp it and snap the slider to match."""
        try:
            v = int(self._smooth_field.text().strip())
        except ValueError:
            v = self._smooth.value()
        v = max(self._smooth.minimum(), min(self._smooth.maximum(), v))
        self._smooth_field.setText(str(v))
        if v != self._smooth.value():
            self._smooth.setValue(v)   # fires _on_smooth_slider -> field + refresh

    def _refresh_plot(self):
        log = self._log_cb.isChecked()
        sigma = float(self._smooth.value())
        curves = []
        markers = []
        for i, ch in enumerate(self._channels):
            if i < len(self._hists) and ch["cb"].isChecked():
                h = self._hists[i].astype(np.float32)
                if sigma > 0:                       # Gaussian-smooth the raw counts
                    h = cv2.GaussianBlur(h.reshape(-1, 1), (0, 0), sigma).flatten()
                if log:
                    h = np.log1p(h)
                peak = float(h.max())
                norm = h / peak if peak > 0 else h
                curves.append((ch["color"], norm, ch["vmax"]))
                lo, hi = ch["slider"].values()
                markers.append((ch["color"], lo, hi, ch["vmax"], ch["name"]))
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
            self._img_h, self._img_w = image.shape[:2]   # full size drives coords
            # The on-screen view is only a few hundred px; cap the display pixmap
            # so switching to a huge image doesn't build a full-res QImage. The
            # pixel-readout (NeighborhoodPanel) and histogram still use full res.
            self._pixmap = QtGui.QPixmap.fromImage(cv_to_qimage(downscale_max(image, 2048)))
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


def _hue_ring_colors(n: int = 72):
    """The true sRGB colours around the CIELAB hue circle (L=60, fixed chroma) — so the
    equator ring shows which colour sits at each hue. Computed once at import."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lab8 = np.zeros((1, n, 3), np.uint8)
    lab8[0, :, 0] = int(round(60.0 * 255 / 100))          # L
    lab8[0, :, 1] = np.clip(60.0 * np.cos(ang) + 128, 0, 255)   # a
    lab8[0, :, 2] = np.clip(60.0 * np.sin(ang) + 128, 0, 255)   # b
    bgr = cv2.cvtColor(lab8, cv2.COLOR_Lab2BGR)[0]
    return [QtGui.QColor(int(r), int(g), int(b)) for b, g, r in bgr]


_HUE_RING = _hue_ring_colors()


class ClusterScatter3D(QtWidgets.QWidget):
    """Interactive 3-D scatter of clustered colours in **CIELAB** space: each pixel placed at
    its (L, a, b), with a faint reference colour sphere, a hue-rainbow **equator** (showing
    where each hue sits in the a–b plane), and the neutral **L** (gray) axis up the middle.
    **Drag to rotate, scroll to zoom.** Toggles: per-cluster transparent enclosing spheres,
    and painting points in their **true colour** instead of the cluster mean. For clustering
    nodes the inspector shows this in place of the pixel grid + histogram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setToolTip("Drag to rotate · scroll to zoom — the CIELAB colour space of the clusters")
        self._pts = None            # (N, 3) display coords (x=a, y=L, z=b), centred & scaled
        self._qmean = []            # QColor per point: its cluster's mean colour
        self._qtrue = []            # QColor per point: its own original colour
        self._spheres = []          # [(centre3, radius, QColor), ...] per cluster
        self._R, self._Req, self._scale, self._Lc = 0.6, 0.5, 1.0, 50.0
        self._yaw, self._pitch, self._zoom = 0.7, 0.45, 1.0
        self._last = None
        # Toggles float over the top-left corner.
        css = "QCheckBox{color:#eee; background:rgba(15,15,15,180); padding:2px 5px; border-radius:3px;}"
        self._spheres_cb = QtWidgets.QCheckBox("spheres", self)
        self._spheres_cb.move(6, 6); self._spheres_cb.setStyleSheet(css)
        self._spheres_cb.setChecked(True)       # on by default
        self._spheres_cb.setToolTip(
            "Per-cluster enclosing spheres.\n"
            "Centre = the cluster's mean CIELAB position (its pixels' average L, a, b).\n"
            "Radius = the 85th-percentile distance of its pixels from that centre — a robust\n"
            "extent that wraps most of the cluster while ignoring stray outliers.")
        self._spheres_cb.toggled.connect(self.update)
        self._true_cb = QtWidgets.QCheckBox("true colour", self)
        self._true_cb.move(6, 30); self._true_cb.setStyleSheet(css)
        self._true_cb.setToolTip("Paint cluster MEMBERS in their own original colour instead "
                                 "of the cluster mean. (Noise is governed by 'flag noise'.)")
        self._true_cb.toggled.connect(self.update)
        # Only meaningful when the payload has noise (-1) points, e.g. Detect Color Centers
        # pixels outside every peak's basin. Hidden otherwise.
        self._flag_cb = QtWidgets.QCheckBox("flag noise", self)
        self._flag_cb.move(6, 54); self._flag_cb.setStyleSheet(css)
        self._flag_cb.setToolTip(
            "Noise = pixels not inside any detected peak's basin.\n"
            "Off: their own (original) colour. On: a magenta flag — this wins for noise\n"
            "even with 'true colour' on.")
        self._flag_cb.setVisible(False)
        self._flag_cb.toggled.connect(self.update)
        self._has_noise = False

    def _to_disp(self, lab):
        """(L, a, b) -> display (x=a, y=L-up, z=b), neutral axis (a=b=0) at the centre line
        and L centred at its mid, uniformly scaled so hues map to angles and spheres stay round."""
        a = np.asarray(lab, np.float32)
        return np.stack([a[..., 1] / self._scale,
                         (a[..., 0] - self._Lc) / self._scale,
                         a[..., 2] / self._scale], axis=-1)

    def set_data(self, pts, labels, centers, sphere_centers=None, sphere_radii=None, point_bgr=None):
        if pts is None or len(pts) == 0 or centers is None or len(centers) == 0:
            self._pts = None
            self.update()
            return
        p = np.asarray(pts, np.float32)                       # (N, 3) = L, a, b
        L, A, B = p[:, 0], p[:, 1], p[:, 2]
        self._Lc = float((L.min() + L.max()) / 2.0)
        half = max(float(np.abs(A).max()), float(np.abs(B).max()),
                   float(np.abs(L - self._Lc).max()), 1e-3)
        self._scale = 2.0 * half                              # uniform: keeps the gray axis at a=b=0
        self._pts = self._to_disp(p)

        cen = np.clip(np.asarray(centers), 0, 255).astype(int)
        raw = np.asarray(labels, np.int64)
        self._has_noise = bool((raw < 0).any())
        self._flag_cb.setVisible(self._has_noise)
        clip = np.clip(raw, 0, len(cen) - 1)
        mean_cols = [QtGui.QColor(int(r), int(g), int(b)) for b, g, r in cen[clip]]
        # Each point's own original colour (for true-colour mode and unflagged noise).
        if point_bgr is not None and len(point_bgr) == len(p):
            pb = np.clip(np.asarray(point_bgr), 0, 255).astype(int)
            self._qtrue = [QtGui.QColor(int(r), int(g), int(b)) for b, g, r in pb]
        else:
            self._qtrue = list(mean_cols)
        # The two toggles act INDEPENDENTLY: "true colour" governs MEMBERS (mean vs their
        # own colour); "flag noise" governs NOISE (own colour vs a magenta flag) and always
        # wins for noise points, even with true colour on. Four precomputed colour lists, one
        # per (true, flag) combo, so paint just picks one. Members are identical across the
        # noise-only variants, so non-CENTERS scatters (no -1) are unaffected.
        flag = QtGui.QColor(200, 0, 200)
        self._qmean, self._qflag, self._qtrueflag = [], [], []
        for i in range(len(raw)):
            noise = raw[i] < 0
            self._qmean.append(self._qtrue[i] if noise else mean_cols[i])       # true off, flag off
            self._qflag.append(flag if noise else mean_cols[i])                 # true off, flag on
            self._qtrueflag.append(flag if noise else self._qtrue[i])           # true on,  flag on

        rad = np.sqrt((self._pts ** 2).sum(axis=1))
        self._R = float(rad.max()) * 1.05 if rad.size else 0.6
        chroma = np.sqrt(self._pts[:, 0] ** 2 + self._pts[:, 2] ** 2)
        self._Req = float(chroma.max()) * 1.08 if chroma.size else 0.5

        self._spheres = []
        if sphere_centers is not None and sphere_radii is not None and len(sphere_centers):
            sc = self._to_disp(np.asarray(sphere_centers, np.float32))
            sr = np.asarray(sphere_radii, np.float32) / self._scale
            for c in range(len(sc)):
                if sr[c] > 0 and c < len(cen):
                    b, g, r = cen[c]
                    self._spheres.append((sc[c], float(sr[c]), QtGui.QColor(int(r), int(g), int(b))))
        self.update()

    def _rot(self, p):
        """Rotate display coords by the current yaw/pitch; return screen-x, screen-y, depth."""
        ca, sa = np.cos(self._yaw), np.sin(self._yaw)
        cb, sb = np.cos(self._pitch), np.sin(self._pitch)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        xr = x * ca + z * sa
        zr = -x * sa + z * ca
        yr = y * cb - zr * sb
        zd = y * sb + zr * cb
        return xr, yr, zd

    def paintEvent(self, _e):
        qp = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        # Gray gradient backdrop matching the L axis: light at the top (high L), dark at the
        # bottom (low L) — so the scene reads as "lightness increases upward".
        grad = QtGui.QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QtGui.QColor(120, 120, 120))   # light top  (high L / white)
        grad.setColorAt(1.0, QtGui.QColor(30, 30, 30))      # dark bottom (low L / black)
        qp.fillRect(self.rect(), QtGui.QBrush(grad))
        if self._pts is None:
            qp.setPen(QtGui.QColor(150, 150, 150))
            qp.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "no cluster data")
            return
        qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        s = self._zoom * min(w, h) * 0.56
        cx, cy = w / 2.0, h / 2.0

        def screen(p3):
            xr, yr, zd = self._rot(np.atleast_2d(p3))
            return cx + xr * s, cy - yr * s, zd

        # Very light transparent reference sphere (its orthographic silhouette is a circle).
        qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 30), 1))
        qp.setBrush(QtGui.QColor(255, 255, 255, 8))
        qp.drawEllipse(QtCore.QPointF(cx, cy), self._R * s, self._R * s)

        # The CIELAB axes as gradient lines: L black->white up the middle, a green->red,
        # b blue->yellow across the equatorial plane (a QPen with a gradient brush).
        def grad_axis(p0, p1, c0, c1, width):
            (x0, y0, _), (x1, y1, _) = screen(p0), screen(p1)
            x0, y0, x1, y1 = float(x0[0]), float(y0[0]), float(x1[0]), float(y1[0])
            g = QtGui.QLinearGradient(x0, y0, x1, y1)
            g.setColorAt(0.0, c0); g.setColorAt(1.0, c1)
            qp.setPen(QtGui.QPen(QtGui.QBrush(g), width))
            qp.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
            return x1, y1                                    # the +tip, for the label

        R, Rq = self._R, self._Req
        l_tip = grad_axis([0, -R, 0], [0, R, 0], QtGui.QColor(0, 0, 0), QtGui.QColor(255, 255, 255), 3)
        a_tip = grad_axis([-Rq, 0, 0], [Rq, 0, 0], QtGui.QColor(0, 170, 70), QtGui.QColor(225, 45, 45), 2)
        b_tip = grad_axis([0, 0, -Rq], [0, 0, Rq], QtGui.QColor(45, 70, 225), QtGui.QColor(220, 200, 0), 2)

        # Hue-rainbow equator in the a–b plane (true colour at each hue).
        n = len(_HUE_RING)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([Rq * np.cos(ang), np.zeros(n), Rq * np.sin(ang)], axis=1)
        rx, ry, _ = screen(ring.astype(np.float32))
        for i in range(n):
            j = (i + 1) % n
            qp.setPen(QtGui.QPen(_HUE_RING[i], 2))
            qp.drawLine(QtCore.QPointF(rx[i], ry[i]), QtCore.QPointF(rx[j], ry[j]))

        # Axis labels at the +tips.
        qp.setPen(QtGui.QColor(245, 245, 245))
        for name, (tx, ty) in (("L", l_tip), ("a", a_tip), ("b", b_tip)):
            qp.drawText(QtCore.QPointF(tx + 3, ty + 4), name)

        # Points, depth-sorted. Independent toggles: true colour -> members in their own
        # colour (else mean); flag noise -> noise magenta (else its own colour). Flag noise
        # always wins for noise, so it overrides true colour there.
        true, flag = self._true_cb.isChecked(), self._flag_cb.isChecked()
        cols = (self._qtrueflag if (true and flag)
                else self._qtrue if true
                else self._qflag if flag
                else self._qmean)
        xr, yr, zd = self._rot(self._pts)
        px, py = cx + xr * s, cy - yr * s
        qp.setPen(QtCore.Qt.PenStyle.NoPen)
        for i in np.argsort(zd):
            qp.setBrush(cols[i])
            qp.drawEllipse(QtCore.QPointF(float(px[i]), float(py[i])), 2.2, 2.2)

        # Per-cluster transparent enclosing spheres (toggle).
        if self._spheres_cb.isChecked():
            for centre3, rad, qcol in self._spheres:
                sx, sy, _ = screen(centre3)
                fill = QtGui.QColor(qcol); fill.setAlpha(38)
                edge = QtGui.QColor(qcol); edge.setAlpha(165)
                qp.setBrush(fill)
                qp.setPen(QtGui.QPen(edge, 1))
                qp.drawEllipse(QtCore.QPointF(float(sx[0]), float(sy[0])), rad * s, rad * s)

    def mousePressEvent(self, e):
        self._last = e.position()

    def mouseReleaseEvent(self, _e):
        self._last = None

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        d = e.position() - self._last
        self._yaw += d.x() * 0.01
        self._pitch = max(-1.4, min(1.4, self._pitch + d.y() * 0.01))
        self._last = e.position()
        self.update()

    def wheelEvent(self, e):
        self._zoom = max(0.4, min(4.0, self._zoom * (1.0 + e.angleDelta().y() / 1200.0)))
        self.update()


# ---------------------------------------------------------------------------
# The pane
# ---------------------------------------------------------------------------
class InspectorPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(340)
        self.setMaximumWidth(900)   # generous: the 3-D scatter benefits; fills when a pane collapses

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

        # Pseudocode view — shown only for nodes that expose get_pseudocode()
        # (the Export Code node). Read-only and selectable so it can be copied.
        self._code = QtWidgets.QPlainTextEdit()
        self._code.setReadOnly(True)
        self._code.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._code.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self._code.setVisible(False)
        layout.addWidget(self._code, 1)

        # Image + (pixel grid, histogram) | OR the interactive 3-D cluster scatter, which
        # replaces the grid+histogram for clustering nodes. Thin 1px handles between.
        self._panels = LineSplitter(QtCore.Qt.Orientation.Vertical)
        self._image = ImagePanel()
        self._neigh = NeighborhoodPanel()
        self._hist = HistogramPanel()
        self._scatter = ClusterScatter3D()
        for i, panel in enumerate((self._image, self._neigh, self._hist, self._scatter)):
            self._panels.addWidget(panel)
            self._panels.setStretchFactor(i, 1)
        self._scatter.setVisible(False)            # only for clustering nodes
        self._panels.setSizes([1100, 950, 950, 1100])
        layout.addWidget(self._panels, 1)

        self._node = None
        self._disp = None       # uint8 display image (unfiltered, native channels)
        self._chan_img = None   # the BGR/HSL view the histogram + filter operate on
        self._vmaxes = []       # per-channel raw value max (Hue -> 179)
        self._channels = 0
        self._names = []
        self._frozen = False

        self._image.setToolTip("Left-click: freeze the pixel readout · Right-click: release")
        self._image.pixelHovered.connect(self._on_hover)
        self._image.pixelClicked.connect(self._on_click)
        self._image.pixelReleased.connect(self._on_release)
        self._hist.rangesChanged.connect(self._apply_filter)
        self._hist.viewChanged.connect(lambda: self._recompute(reset=True))

    # --- public API --------------------------------------------------------
    def set_node(self, node) -> None:
        self._node = node
        self._frozen = False
        if node is None:
            name = "(no selection)"
        else:
            op = getattr(node, "op", None)
            name = op.label if op is not None else "Image"   # display label, not op id
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

    def _update_code(self) -> bool:
        """Show the node's pseudocode (Export Code node) or hide the code view.
        Returns True when the code view is showing — the caller then suppresses the
        image/histogram (the Export Code node carries a pass-through image, but it's
        a text node: only the pseudocode is meaningful)."""
        getter = getattr(self._node, "get_pseudocode", None) if self._node is not None else None
        showing = callable(getter)
        if showing:
            try:
                self._code.setPlainText(getter())
            except Exception as e:  # noqa: BLE001
                self._code.setPlainText(f"# {e}")
        self._code.setVisible(showing)
        return showing

    def _recompute(self, reset: bool) -> None:
        code_only = self._update_code()
        # Text-only nodes (Export Code): hide image, pixel grid, histogram, scatter
        # and the size/type metadata — the pseudocode view is the whole inspector.
        self._panels.setVisible(not code_only)
        self._meta.setVisible(not code_only)
        self._update_frame_controls()
        if code_only:
            return
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
        self._names = channel_names(self._node, channels)   # native — for the pixel readout
        h, w = disp.shape[:2]
        self._meta.setText(f"{w}×{h}   {self._type_text(self._node, channels)}   {channels} ch")

        # Chart previews (cluster palettes, the Histogram node, …) are plotted graphs: a
        # pixel-neighbourhood grid and a per-channel histogram of them are meaningless, so
        # hide both. A clustering payload additionally carries a 3-D colour scatter to show
        # in their place (drag-rotate / zoom).
        chart = getattr(getattr(self._node, "op", None), "preview_is_chart", False)
        self._neigh.setVisible(not chart)
        self._hist.setVisible(not chart)

        payload = self._node.get_output_image()
        scat = (payload["diag"] if isinstance(payload, dict)
                and isinstance(payload.get("diag"), dict)
                and "scatter_lab" in payload["diag"] else None)
        self._scatter.setVisible(scat is not None)
        if scat is not None:
            colours = scat.get("centers")            # CENTERS payloads carry seed colours here
            if colours is None:
                colours = payload.get("centers")     # CLUSTERS payloads use the top-level key
            self._scatter.set_data(scat["scatter_lab"], scat.get("scatter3d_labels"),
                                   colours, scat.get("cluster_centers_lab"),
                                   scat.get("cluster_radii_lab"), point_bgr=scat.get("scatter3d"))

        if chart:
            self._hist.clear()
            self._channels = 0
            self._image.set_image(disp)        # the rendered chart preview, unfiltered
            if reset:
                self._neigh.reset_center()
            return

        # Image node: the histogram + filter operate on the chosen BGR/HSL view.
        self._chan_img, view_names, self._vmaxes = self._channel_view(disp, channels)
        nch = len(view_names)
        if reset or nch != self._channels:
            self._hist.configure(nch, view_names)
            self._channels = nch
        self._hist.set_hists(self._compute_hists(self._chan_img, self._vmaxes))
        if reset:
            self._neigh.reset_center()
        self._apply_filter()   # updates both the image and the neighbourhood

    def _native_space(self) -> str:
        return (getattr(getattr(self._node, "gnode", None), "color_space", "") or "").lower()

    def _to_bgr(self, disp):
        sp = self._native_space()
        try:
            if sp == "hls":
                return cv2.cvtColor(disp, cv2.COLOR_HLS2BGR)
            if sp == "hsv":
                return cv2.cvtColor(disp, cv2.COLOR_HSV2BGR)
            if sp == "lab":
                return cv2.cvtColor(disp, cv2.COLOR_Lab2BGR)
        except cv2.error:
            pass
        return disp   # already BGR / unknown 3-channel

    def _channel_view(self, disp, channels):
        """(chan_img, names, vmaxes) the histogram + filter use, per the BGR/HSL
        toggle. Grayscale ignores the toggle."""
        if channels == 1:
            return disp, ["Gray"], [255]
        bgr = self._to_bgr(disp)
        if self._hist.color_view() == "hsl":
            hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)   # OpenCV order: H, L, S
            names = ["H", "L", "S"]
            return hls, names, [_ch_vmax(n) for n in names]
        names = ["B", "G", "R"]
        return bgr, names, [_ch_vmax(n) for n in names]

    @staticmethod
    def _compute_hists(img, vmaxes):
        n = 1 if img.ndim == 2 else img.shape[2]
        return [cv2.calcHist([img], [c], None, [vmaxes[c] + 1], [0, vmaxes[c] + 1]).flatten()
                for c in range(n)]

    def _filtered_image(self):
        """The (native) display image with the histogram ranges masked out — pixels
        whose chosen-view channels fall outside an active range are set to black."""
        if self._disp is None or self._chan_img is None:
            return None
        chan = self._chan_img
        mask = np.ones(self._disp.shape[:2], dtype=bool)
        for i, (enabled, lo, hi) in enumerate(self._hist.ranges()):
            vm = self._vmaxes[i] if i < len(self._vmaxes) else 255
            if not enabled or (lo <= 0 and hi >= vm):
                continue
            ch = chan if chan.ndim == 2 else chan[:, :, i]
            mask &= (ch >= lo) & (ch <= hi)
        if mask.all():
            return self._disp
        out = self._disp.copy()
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
