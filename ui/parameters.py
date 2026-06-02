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
    LABEL_W = 84    # fixed label column so controls line up across rows
    SLIDER_MIN = 72  # sliders never shrink below this (so they stay usable)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)   # tight inter-row spacing
        self._node = None
        self._loading = False        # suppress callbacks while populating
        self._suppress_slider = False  # set while a typed value snaps the slider

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

    def _value_field(self, text: str) -> QtWidgets.QLineEdit:
        """An editable value field shown next to a slider (type an exact value)."""
        f = QtWidgets.QLineEdit(text)
        f.setMinimumWidth(42)    # can shrink to ~5 chars (give the slider room)
        f.setMaximumWidth(72)    # but a log value like "1.000.000" still fits
        f.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return f

    def _add_slider_row(self, spec, value, *, pos_min, pos_max, to_val, to_pos,
                        fmt, parse, live, odd=False) -> None:
        """A label + slider + editable value field. ``to_val``/``to_pos`` map
        between the slider position and the parameter value; ``fmt``/``parse``
        between the value and the field text. Dragging the slider commits on
        release (or live for every step if ``live``); typing in the field commits
        the exact value immediately and snaps the slider to the nearest position."""
        row = self._row(spec)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(pos_min, pos_max)
        slider.setMinimumWidth(self.SLIDER_MIN)      # stays usable in a narrow panel
        slider.setValue(to_pos(value))               # set before connecting
        field = self._value_field(fmt(value))
        state = {"v": value}

        def commit(v):
            state["v"] = v
            self._commit(spec.name, v, commit=True)

        def on_slider(pos):
            if self._suppress_slider:
                return
            if odd and pos % 2 == 0:
                slider.setValue(pos + 1 if pos < pos_max else pos - 1)
                return
            v = to_val(pos)
            field.setText(fmt(v))
            if live or not slider.isSliderDown():
                commit(v)

        def on_field():
            v = parse(field.text())
            if v is None:                            # invalid -> revert
                field.setText(fmt(state["v"]))
                return
            if v == state["v"]:                      # unchanged -> no recompute
                field.setText(fmt(v))
                return
            self._suppress_slider = True             # move slider without re-committing
            slider.setValue(to_pos(v))
            self._suppress_slider = False
            field.setText(fmt(v))
            commit(v)                                # commit the *exact* typed value

        slider.valueChanged.connect(on_slider)
        slider.sliderReleased.connect(lambda: commit(to_val(slider.value())))
        field.editingFinished.connect(on_field)
        row.addWidget(slider, 1)
        row.addWidget(field)

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

    @staticmethod
    def _fmt_thousands(v) -> str:
        # Thousands separator is a dot (e.g. 1.000.000), no decimals.
        return f"{int(v):,}".replace(",", ".")

    def _add_int(self, spec, value) -> None:
        lo = int(spec.min if spec.min is not None else 0)
        hi = int(spec.max if spec.max is not None else 100)
        if getattr(spec, "log", False):
            self._add_log_int(spec, value, lo, hi)
            return

        def parse(t):
            try:
                return max(lo, min(hi, int(t.strip())))
            except (ValueError, AttributeError):
                return None

        self._add_slider_row(
            spec, int(value), pos_min=lo, pos_max=hi,
            to_val=lambda p: int(p),
            to_pos=lambda v: max(lo, min(hi, int(round(v)))),
            fmt=lambda v: str(int(v)), parse=parse,
            live=getattr(spec, "live", False), odd=bool(spec.odd))

    def _add_log_int(self, spec, value, lo, hi) -> None:
        """Integer slider with a logarithmic response: fine control near the low
        end (small features), coarse near the top. For wide-range area filters."""
        lo_eff = max(1, lo)               # log needs a positive lower bound
        hi_eff = max(lo_eff + 1, hi)
        ln_lo, ln_hi = math.log(lo_eff), math.log(hi_eff)
        steps = 1000

        def to_val(pos):
            return max(lo, min(hi, int(round(math.exp(ln_lo + (pos / steps) * (ln_hi - ln_lo))))))

        def to_pos(v):
            v = max(lo_eff, min(hi_eff, int(v)))
            return int(round(steps * (math.log(v) - ln_lo) / (ln_hi - ln_lo)))

        def parse(t):
            try:                          # accept dotted thousands ("1.000.000")
                return max(lo, min(hi, int(t.replace(".", "").replace(" ", "").strip())))
            except (ValueError, AttributeError):
                return None

        self._add_slider_row(
            spec, int(value), pos_min=0, pos_max=steps,
            to_val=to_val, to_pos=to_pos,
            fmt=self._fmt_thousands, parse=parse,
            live=getattr(spec, "live", False))

    def _add_float(self, spec, value) -> None:
        lo = float(spec.min if spec.min is not None else 0.0)
        hi = float(spec.max if spec.max is not None else 1.0)
        step = float(spec.step or 0.01)
        steps = max(1, int(round((hi - lo) / step)))

        def parse(t):
            try:
                return max(lo, min(hi, float(t.strip())))
            except (ValueError, AttributeError):
                return None

        self._add_slider_row(
            spec, float(value), pos_min=0, pos_max=steps,
            to_val=lambda pos: lo + pos * step,
            to_pos=lambda v: max(0, min(steps, int(round((float(v) - lo) / step)))),
            fmt=lambda v: f"{float(v):.2f}", parse=parse,
            live=getattr(spec, "live", False))

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
