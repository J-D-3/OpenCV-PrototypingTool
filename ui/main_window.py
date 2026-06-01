"""The application main window: registry-driven sidebar, canvas, and the
parameter panel (frontend)."""
import cv2
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from core.operations import by_label, ops_by_category, CATEGORY_ORDER
from ui.nodes import Node, ImageNode, FunctionNode, SaveToFileNode
from ui.canvas import ImageDropWidget, DEFAULT_ICON_SIZE
from ui.viewer import ImageViewerWindow
from ui.parameters import ParameterPanel

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_image_bgr: Optional[any], window_title: str):
        super().__init__()
        self.setWindowTitle(window_title)
        self.resize(1500, 1000)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Sidebar
        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(300)
        sidebar.setLayout(QtWidgets.QVBoxLayout())
        sidebar.layout().setContentsMargins(12, 12, 12, 12)
        sidebar.layout().setSpacing(8)

        title = QtWidgets.QLabel("Menu")
        title.setStyleSheet("font-weight: bold;")
        open_btn = QtWidgets.QPushButton("Open ImageÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦")
        size_label = QtWidgets.QLabel(f"Icon size: {DEFAULT_ICON_SIZE} px")
        size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        size_slider.setRange(24, 256)
        size_slider.setSingleStep(1)
        size_slider.setPageStep(4)
        size_slider.setValue(DEFAULT_ICON_SIZE)
        size_slider.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)

        # OpenCV functions tree
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        tree.setAnimated(True)
        tree.setMinimumHeight(300)

        for cat in CATEGORY_ORDER:
            cat_item = QtWidgets.QTreeWidgetItem([cat])
            cat_item.setFlags(cat_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            tree.addTopLevelItem(cat_item)
            for op in ops_by_category.get(cat, []):
                op_item = QtWidgets.QTreeWidgetItem([op.label])
                op_item.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                                {"name": op.id, "in": op.in_label, "out": op.out_label})
                cat_item.addChild(op_item)

        tree.expandAll()

        # Info box at bottom
        info_group = QtWidgets.QGroupBox("Function info")
        info_layout = QtWidgets.QVBoxLayout()
        info_group.setLayout(info_layout)
        info_name = QtWidgets.QLabel("Select a function")
        info_types = QtWidgets.QLabel("")
        info_types.setStyleSheet("color: #555;")
        info_layout.addWidget(info_name)
        info_layout.addWidget(info_types)
        
        # Parameter controls â€” auto-generated from the selected op's schema.
        param_group = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QVBoxLayout()
        param_group.setLayout(param_layout)
        param_group.setVisible(False)  # Hidden by default
        self.param_panel = ParameterPanel()
        param_layout.addWidget(self.param_panel)

        sidebar.layout().addWidget(title)
        sidebar.layout().addWidget(open_btn)
        sidebar.layout().addWidget(size_label)
        sidebar.layout().addWidget(size_slider)
        sidebar.layout().addSpacing(8)
        sidebar.layout().addSpacing(8)
        sidebar.layout().addWidget(tree)
        sidebar.layout().addSpacing(8)
        sidebar.layout().addWidget(param_group)
        sidebar.layout().addSpacing(8)
        sidebar.layout().addWidget(info_group)
        sidebar.layout().addStretch(1)

        # Main drop/view pane
        self.drop_widget = ImageDropWidget()

        splitter.addWidget(sidebar)
        splitter.addWidget(self.drop_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        container = QtWidgets.QWidget()
        container.setLayout(QtWidgets.QHBoxLayout())
        container.layout().setContentsMargins(8, 8, 8, 8)
        container.layout().addWidget(splitter)
        self.setCentralWidget(container)

        open_btn.clicked.connect(self.drop_widget.browse_for_image)
        self.drop_widget.imageLoaded.connect(self.on_image_loaded)
        
        # Connect double-click signal to open image viewer
        self.drop_widget.view.nodeDoubleClicked.connect(self.open_image_viewer)

        def on_size_changed(value: int) -> None:
            self.drop_widget.icon_size = int(value)
            size_label.setText(f"Icon size: {value} px")
            self.drop_widget.view.set_thumb_size(int(value))
            self.drop_widget.resize_all_icons(int(value))

        size_slider.valueChanged.connect(on_size_changed)

        def on_tree_selection_changed() -> None:
            items = tree.selectedItems()
            if not items:
                info_name.setText("Select a function")
                info_types.setText("")
                return
            item = items[0]
            meta = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(meta, dict):
                info_name.setText(f"{meta.get('name','')}()")
                info_types.setText(f"Input: {meta.get('in','?')}\nOutput: {meta.get('out','?')}")
            else:
                info_name.setText("Select a function")
                info_types.setText("")

        tree.itemSelectionChanged.connect(on_tree_selection_changed)

        def on_tree_item_double_clicked(item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
            meta = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(meta, dict) and 'name' in meta:
                # Add a function node to the scene with the display label
                self.drop_widget.add_function_node(item.text(0), None, meta)
                # Also update info box to reflect the selected function
                info_name.setText(f"{meta.get('name','')}()")
                info_types.setText(f"Input: {meta.get('in','?')}\nOutput: {meta.get('out','?')}")

        tree.itemDoubleClicked.connect(on_tree_item_double_clicked)

        # Update info panel when selecting a function node in the scene
        def on_scene_selection_changed() -> None:
            try:
                selected = self.drop_widget.view._scene.selectedItems()
                if not selected:
                    self.param_panel.clear()
                    param_group.setVisible(False)
                    return
                sel = selected[0]
                if isinstance(sel, FunctionNode):
                    meta = getattr(sel, "_meta", None)
                    if isinstance(meta, dict):
                        info_name.setText(f"{meta.get('name','')}()")
                        info_types.setText(f"Input: {meta.get('in','?')}\nOutput: {meta.get('out','?')}")
                        # Auto-build parameter controls from the op's schema.
                        self.param_panel.set_node(sel)
                        param_group.setVisible(self.param_panel.has_controls())
                elif isinstance(sel, ImageNode):
                    meta = getattr(sel, "_meta", None)
                    if isinstance(meta, dict):
                        info_name.setText("Image")
                        ch = meta.get('channels', '?')
                        info_types.setText(
                            f"Size: {meta.get('w','?')}ÃƒÆ’Ã¢â‚¬â€{meta.get('h','?')}\n"
                            f"Channels: {ch}\n"
                            f"Type: {meta.get('type','?')}"
                        )
                    self.param_panel.clear()
                    param_group.setVisible(False)
            except RuntimeError:
                # Scene or items have been deleted, ignore
                pass

        self.drop_widget.view._scene.selectionChanged.connect(on_scene_selection_changed)

        if initial_image_bgr is not None:
            self.drop_widget.add_icon(initial_image_bgr, None)

    @QtCore.pyqtSlot(Path)
    def on_image_loaded(self, path: Path) -> None:
        self.setWindowTitle(f"Image - {path.name}")

    @QtCore.pyqtSlot(Node)
    def open_image_viewer(self, node: Node) -> None:
        """Open an image viewer window for the specified node."""
        viewer = ImageViewerWindow(node, self)
        viewer.show()
