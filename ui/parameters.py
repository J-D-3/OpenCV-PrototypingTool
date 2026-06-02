"""ParameterPanel — parameter controls generated from a node's schema (frontend).

Given a FunctionNode, it reads ``node.op.params`` (a list of
``core.operations.ParamSpec``) and builds the matching widgets. Sliders emit a
live *preview* while dragging and *commit* on release; discrete controls
(combo / checkbox / line edit) commit immediately. This replaces the former
hand-written per-function control code: a new operation needs no UI work.
"""
import math

from PyQt6 import QtCore, QtWidgets


class ParameterPanel(QtWidgets.QWidget):
    LABEL_W = 96   # fixed label column so controls line up across rows
    VALUE_W = 38   # min width for a slider's value readout (right of the slider)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)   # tight inter-row spacing
        self._node = None
        self._loading = False   # suppress callbacks while populating

    # --- public API --------------------------------------------------------
    def clear(self) -> None:
        self.set_node(None)

    def set_node(self, node) -> None:
        self._clear_widgets()
        self._node = node
        if node is None:
            return
        specs = [s for s in node.op.params if getattr(s, "show", True)]
        if not specs:
            return
        values = node.get_parameters()
        self._loading = True
        try:
            for spec in specs:
                self._add_control(spec, values.get(spec.name, spec.default))
            self._layout.addStretch(1)
        finally:
            self._loading = False

    def has_controls(self) -> bool:
        return self._layout.count() > 0

    # --- internals ---------------------------------------------------------
    def _clear_widgets(self) -> None:
        while self._layout.count():
            child = self._layout.takeAt(0)
            w = child.widget()
            if w is not None:
                w.deleteLater()

    def _commit(self, name, value, commit: bool) -> None:
        if self._loading or self._node is None:
            return
        self._node.set_parameter(name, value, preview_mode=not commit)

    @staticmethod
    def _title(spec) -> str:
        return spec.label or spec.name.replace("_", " ").title()

    def _row(self, spec=None):
        """Add a horizontal row (label on the left) and return its layout so the
        caller can append the control inline."""
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self._layout.addWidget(w)
        if spec is not None:
            lbl = QtWidgets.QLabel(self._title(spec))
            lbl.setFixedWidth(self.LABEL_W)
            lbl.setToolTip(self._title(spec))   # full text if the column clips it
            h.addWidget(lbl)
        return h

    def _value_label(self, text: str) -> QtWidgets.QLabel:
        rd = QtWidgets.QLabel(text)
        rd.setMinimumWidth(self.VALUE_W)
        rd.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return rd

    def _add_control(self, spec, value) -> None:
        kind = spec.kind
        if kind == "int":
            self._add_int(spec, value)
        elif kind == "float":
            self._add_float(spec, value)
        elif kind == "bool":
            self._add_bool(spec, value)
        elif kind in ("enum", "choice"):
            self._add_choice(spec, value)
        else:  # "str" / "path"
            self._add_text(spec, value, browse=(kind == "path"))

    def _add_int(self, spec, value) -> None:
        lo = int(spec.min if spec.min is not None else 0)
        hi = int(spec.max if spec.max is not None else 100)
        if getattr(spec, "log", False):
            self._add_log_int(spec, value, lo, hi)
            return
        row = self._row(spec)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(int(value))
        readout = self._value_label(str(int(value)))

        def on_change(v):
            if spec.odd and v % 2 == 0:
                v = v + 1 if v < hi else v - 1
                slider.setValue(v)   # re-fires on_change with an odd value
                return
            readout.setText(str(v))
            # Only recompute live for non-drag changes (keyboard / click / wheel).
            # A mouse drag updates the label but defers the eval to release.
            if not slider.isSliderDown():
                self._commit(spec.name, v, commit=True)

        slider.valueChanged.connect(on_change)
        slider.sliderReleased.connect(lambda: self._commit(spec.name, slider.value(), commit=True))
        row.addWidget(slider, 1)
        row.addWidget(readout)

    def _add_log_int(self, spec, value, lo, hi) -> None:
        """Integer slider with a logarithmic response: fine control near the low
        end (small features), coarse near the top. For wide-range area filters."""
        row = self._row(spec)
        lo_eff = max(1, lo)               # log needs a positive lower bound
        hi_eff = max(lo_eff + 1, hi)
        ln_lo, ln_hi = math.log(lo_eff), math.log(hi_eff)
        steps = 1000
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, steps)

        def to_val(pos):
            v = int(round(math.exp(ln_lo + (pos / steps) * (ln_hi - ln_lo))))
            return max(lo, min(hi, v))

        def to_pos(v):
            v = max(lo_eff, min(hi_eff, int(v)))
            return int(round(steps * (math.log(v) - ln_lo) / (ln_hi - ln_lo)))

        slider.setValue(to_pos(value))
        readout = self._value_label(f"{int(value):,}")   # show the true value at init

        def on_change(pos):
            v = to_val(pos)
            readout.setText(f"{v:,}")
            if not slider.isSliderDown():
                self._commit(spec.name, v, commit=True)

        slider.valueChanged.connect(on_change)
        slider.sliderReleased.connect(lambda: self._commit(spec.name, to_val(slider.value()), commit=True))
        row.addWidget(slider, 1)
        row.addWidget(readout)

    def _add_float(self, spec, value) -> None:
        row = self._row(spec)
        lo = float(spec.min if spec.min is not None else 0.0)
        hi = float(spec.max if spec.max is not None else 1.0)
        step = float(spec.step or 0.01)
        steps = max(1, int(round((hi - lo) / step)))
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, steps)
        slider.setValue(max(0, min(steps, int(round((float(value) - lo) / step)))))
        readout = self._value_label(f"{float(value):.2f}")

        def to_val(pos):
            return lo + pos * step

        def on_change(pos):
            val = to_val(pos)
            readout.setText(f"{val:.2f}")
            if not slider.isSliderDown():
                self._commit(spec.name, val, commit=True)

        slider.valueChanged.connect(on_change)
        slider.sliderReleased.connect(lambda: self._commit(spec.name, to_val(slider.value()), commit=True))
        row.addWidget(slider, 1)
        row.addWidget(readout)

    def _add_bool(self, spec, value) -> None:
        # The checkbox carries its own label, so it sits on one line already.
        row = self._row()
        cb = QtWidgets.QCheckBox(self._title(spec))
        cb.setChecked(bool(value))
        cb.toggled.connect(lambda checked: self._commit(spec.name, checked, commit=True))
        row.addWidget(cb, 1)

    def _add_choice(self, spec, value) -> None:
        row = self._row(spec)
        combo = QtWidgets.QComboBox()
        select = 0
        for i, (label, val) in enumerate(spec.choices or []):
            combo.addItem(label, val)
            if val == value:
                select = i
        combo.setCurrentIndex(select)
        combo.currentIndexChanged.connect(
            lambda _i: self._commit(spec.name, combo.currentData(), commit=True))
        row.addWidget(combo, 1)

    def _add_text(self, spec, value, browse: bool = False) -> None:
        row = self._row(spec)
        edit = QtWidgets.QLineEdit(str(value or ""))
        edit.editingFinished.connect(lambda: self._commit(spec.name, edit.text(), commit=True))
        row.addWidget(edit, 1)
        if browse:
            btn = QtWidgets.QPushButton("…")
            btn.setFixedWidth(28)

            def on_browse():
                fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save Image As", "./output/",
                    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)")
                if fn:
                    edit.setText(fn)
                    self._commit(spec.name, fn, commit=True)

            btn.clicked.connect(on_browse)
            row.addWidget(btn)
