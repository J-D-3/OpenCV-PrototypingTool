"""ParameterPanel — parameter controls generated from a node's schema (frontend).

Given a FunctionNode, it reads ``node.op.params`` (a list of
``core.operations.ParamSpec``) and builds the matching widgets. Sliders emit a
live *preview* while dragging and *commit* on release; discrete controls
(combo / checkbox / line edit) commit immediately. This replaces the former
hand-written per-function control code: a new operation needs no UI work.
"""
from PyQt6 import QtCore, QtWidgets


class ParameterPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
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
        self._layout.addWidget(QtWidgets.QLabel(self._title(spec) + ":"))
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        lo = int(spec.min if spec.min is not None else 0)
        hi = int(spec.max if spec.max is not None else 100)
        slider.setRange(lo, hi)
        slider.setValue(int(value))
        readout = QtWidgets.QLabel(str(int(value)))

        def on_change(v):
            if spec.odd and v % 2 == 0:
                v = v + 1 if v < hi else v - 1
                slider.setValue(v)   # re-fires on_change with an odd value
                return
            readout.setText(str(v))
            self._commit(spec.name, v, commit=False)

        slider.valueChanged.connect(on_change)
        slider.sliderReleased.connect(lambda: self._commit(spec.name, slider.value(), commit=True))
        self._layout.addWidget(slider)
        self._layout.addWidget(readout)

    def _add_float(self, spec, value) -> None:
        self._layout.addWidget(QtWidgets.QLabel(self._title(spec) + ":"))
        lo = float(spec.min if spec.min is not None else 0.0)
        hi = float(spec.max if spec.max is not None else 1.0)
        step = float(spec.step or 0.01)
        steps = max(1, int(round((hi - lo) / step)))
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, steps)
        slider.setValue(max(0, min(steps, int(round((float(value) - lo) / step)))))
        readout = QtWidgets.QLabel(f"{float(value):.2f}")

        def to_val(pos):
            return lo + pos * step

        def on_change(pos):
            val = to_val(pos)
            readout.setText(f"{val:.2f}")
            self._commit(spec.name, val, commit=False)

        slider.valueChanged.connect(on_change)
        slider.sliderReleased.connect(lambda: self._commit(spec.name, to_val(slider.value()), commit=True))
        self._layout.addWidget(slider)
        self._layout.addWidget(readout)

    def _add_bool(self, spec, value) -> None:
        cb = QtWidgets.QCheckBox(self._title(spec))
        cb.setChecked(bool(value))
        cb.toggled.connect(lambda checked: self._commit(spec.name, checked, commit=True))
        self._layout.addWidget(cb)

    def _add_choice(self, spec, value) -> None:
        self._layout.addWidget(QtWidgets.QLabel(self._title(spec) + ":"))
        combo = QtWidgets.QComboBox()
        select = 0
        for i, (label, val) in enumerate(spec.choices or []):
            combo.addItem(label, val)
            if val == value:
                select = i
        combo.setCurrentIndex(select)
        combo.currentIndexChanged.connect(
            lambda _i: self._commit(spec.name, combo.currentData(), commit=True))
        self._layout.addWidget(combo)

    def _add_text(self, spec, value, browse: bool = False) -> None:
        self._layout.addWidget(QtWidgets.QLabel(self._title(spec) + ":"))
        edit = QtWidgets.QLineEdit(str(value or ""))
        edit.editingFinished.connect(lambda: self._commit(spec.name, edit.text(), commit=True))
        self._layout.addWidget(edit)
        if browse:
            btn = QtWidgets.QPushButton("Browse...")

            def on_browse():
                fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save Image As", "./output/",
                    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)")
                if fn:
                    edit.setText(fn)
                    self._commit(spec.name, fn, commit=True)

            btn.clicked.connect(on_browse)
            self._layout.addWidget(btn)
