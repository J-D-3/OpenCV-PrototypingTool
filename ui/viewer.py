"""Standalone window for inspecting a node's result (frontend).

Signal-driven (no polling): it subscribes to the node's ``outputChanged`` signal
and refreshes when the backend recomputes. It shows the node's *preview* image
(an op may render one — e.g. contours drawn onto the input) plus a summary of
key facts the op exposes (e.g. number of contours).
"""
from PyQt6 import QtCore, QtGui, QtWidgets

from ui.nodes import Node
from ui.image_utils import cv_to_qimage


class ImageViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, node: Node, parent=None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle(f"Inspector - {node._meta.get('name', 'Node')}")
        self.setMinimumSize(400, 300)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Summary line (key facts the op exposes).
        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setStyleSheet("color: #333; font-weight: bold;")
        self.summary_label.setVisible(False)
        layout.addWidget(self.summary_label)

        # Scrollable image area.
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(100, 100)
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)

        # Zoom controls.
        controls_layout = QtWidgets.QHBoxLayout()
        zoom_out_btn = QtWidgets.QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_in_btn = QtWidgets.QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        fit_btn = QtWidgets.QPushButton("Fit to Window")
        fit_btn.clicked.connect(self.fit_to_window)
        self.zoom_label = QtWidgets.QLabel("100%")
        self.zoom_label.setMinimumWidth(60)
        controls_layout.addWidget(zoom_out_btn)
        controls_layout.addWidget(zoom_in_btn)
        controls_layout.addWidget(fit_btn)
        controls_layout.addWidget(self.zoom_label)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        self.zoom_factor = 1.0

        # Refresh when this node's result changes (no polling).
        self._controller = getattr(node, "controller", None)
        if self._controller is not None:
            self._controller.signals.nodeChanged.connect(self._on_node_changed)
        self.update_image()

    def _on_node_changed(self, qt_node) -> None:
        if qt_node is self.node:
            self.update_image()

    def update_image(self):
        if self.node is None:
            return

        summary = self.node.get_summary()
        if summary:
            self.summary_label.setText("   ".join(f"{k}: {v}" for k, v in summary.items()))
            self.summary_label.setVisible(True)
        else:
            self.summary_label.setVisible(False)

        image = self.node.get_preview_image()
        if image is None:
            self.image_label.setText("No result to display")
            self.image_label.setPixmap(QtGui.QPixmap())
            self.original_pixmap = QtGui.QPixmap()
            return

        self.original_pixmap = QtGui.QPixmap.fromImage(cv_to_qimage(image))
        self.apply_zoom()

    def apply_zoom(self):
        if not hasattr(self, 'original_pixmap') or self.original_pixmap.isNull():
            return
        original_size = self.original_pixmap.size()
        new_size = QtCore.QSize(
            int(original_size.width() * self.zoom_factor),
            int(original_size.height() * self.zoom_factor),
        )
        scaled_pixmap = self.original_pixmap.scaled(
            new_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)
        self.apply_zoom()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
        self.apply_zoom()

    def fit_to_window(self):
        if not hasattr(self, 'original_pixmap') or self.original_pixmap.isNull():
            return
        scroll_area = self.image_label.parent().parent()
        available_size = scroll_area.size()
        original_size = self.original_pixmap.size()
        scale_x = available_size.width() / original_size.width()
        scale_y = available_size.height() / original_size.height()
        self.zoom_factor = min(scale_x, scale_y) * 0.9
        self.apply_zoom()

    def closeEvent(self, event):
        if self._controller is not None:
            try:
                self._controller.signals.nodeChanged.disconnect(self._on_node_changed)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)
