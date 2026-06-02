"""The application main window: registry-driven sidebar, canvas, and the
parameter panel (frontend)."""
import cv2
import json
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from core.operations import by_label, ops_by_category, CATEGORY_ORDER, REGISTRY
from core import codegen
from ui.nodes import Node, ImageNode, FunctionNode, SaveToFileNode
from ui.canvas import ImageDropWidget, DEFAULT_ICON_SIZE
from ui.viewer import ImageViewerWindow
from ui.parameters import ParameterPanel
from ui.inspector_pane import InspectorPane
from ui.widgets import LineSplitter


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_image_bgr: Optional[any], window_title: str):
        super().__init__()
        self.setWindowTitle(window_title)
        self.resize(1500, 1000)

        splitter = LineSplitter(QtCore.Qt.Orientation.Horizontal)

        # Sidebar
        sidebar = QtWidgets.QWidget()
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(300)
        sidebar.setLayout(QtWidgets.QVBoxLayout())
        sidebar.layout().setContentsMargins(12, 12, 12, 12)
        sidebar.layout().setSpacing(8)

        title = QtWidgets.QLabel("Menu")
        title.setStyleSheet("font-weight: bold;")
        sp = QtWidgets.QStyle.StandardPixmap

        def _tool(icon, tip):
            b = QtWidgets.QToolButton()
            b.setIcon(self.style().standardIcon(icon))
            b.setIconSize(QtCore.QSize(22, 22))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            return b

        open_btn = _tool(sp.SP_FileIcon, "Open Image")
        open_imgs_btn = _tool(sp.SP_DirIcon, "Open Images (batch)")
        save_btn = _tool(sp.SP_DialogSaveButton, "Save Pipeline")
        load_btn = _tool(sp.SP_DialogOpenButton, "Load Pipeline")
        toolbar = QtWidgets.QHBoxLayout()
        for _b in (open_btn, open_imgs_btn, save_btn, load_btn):
            toolbar.addWidget(_b)
        toolbar.addStretch()

        # Search bar — filters the tree by display name, category, or the cv::
        # functions a node calls.
        search = QtWidgets.QLineEdit()
        search.setPlaceholderText("Search name / category / cv:: call…")
        search.setClearButtonEnabled(True)

        # OpenCV functions tree
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        tree.setAnimated(True)
        tree.setMinimumHeight(140)

        for cat in CATEGORY_ORDER:
            cat_item = QtWidgets.QTreeWidgetItem([cat])
            cat_item.setFlags(cat_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            tree.addTopLevelItem(cat_item)
            for op in ops_by_category.get(cat, []):
                op_item = QtWidgets.QTreeWidgetItem([op.label])
                # Precomputed haystack: display name + id + category + port labels +
                # the cv:: calls the op makes (so e.g. "gaussianblur" finds Blur).
                haystack = " ".join(
                    [op.label, op.id, cat, op.in_label, op.out_label]
                    + codegen.op_cv_calls(op)).lower()
                op_item.setData(0, QtCore.Qt.ItemDataRole.UserRole,
                                {"name": op.id, "in": op.in_label, "out": op.out_label,
                                 "desc": codegen.op_description(op), "search": haystack})
                op_item.setToolTip(0, codegen.op_description(op))
                cat_item.addChild(op_item)

        tree.expandAll()

        def filter_tree(text: str) -> None:
            q = text.strip().lower()
            for i in range(tree.topLevelItemCount()):
                cat_item = tree.topLevelItem(i)
                cat_match = q in cat_item.text(0).lower()
                any_visible = False
                for j in range(cat_item.childCount()):
                    op_item = cat_item.child(j)
                    meta = op_item.data(0, QtCore.Qt.ItemDataRole.UserRole) or {}
                    visible = (not q) or cat_match or (q in meta.get("search", ""))
                    op_item.setHidden(not visible)
                    any_visible = any_visible or visible
                # Hide an empty category while searching; keep it if its name matched.
                cat_item.setHidden(bool(q) and not any_visible and not cat_match)
                if q:
                    cat_item.setExpanded(True)
            if not q:
                tree.expandAll()

        search.textChanged.connect(filter_tree)
        self.func_tree = tree
        self.func_search = search

        # Info box at bottom — fixed height, scrollable when the description is long.
        info_group = QtWidgets.QGroupBox("Function info")
        info_group.setFixedHeight(120)
        info_outer = QtWidgets.QVBoxLayout(info_group)
        info_outer.setContentsMargins(4, 4, 4, 4)
        info_scroll = QtWidgets.QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        info_content = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_content)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_scroll.setWidget(info_content)
        info_outer.addWidget(info_scroll)
        info_name = QtWidgets.QLabel("Select a function")
        info_types = QtWidgets.QLabel("")
        info_types.setStyleSheet("color: #555;")
        info_desc = QtWidgets.QLabel("")
        info_desc.setStyleSheet("color: #444;")
        info_desc.setWordWrap(True)
        info_layout.addWidget(info_name)
        info_layout.addWidget(info_types)
        info_layout.addWidget(info_desc)
        info_layout.addStretch(1)

        def set_func_info(op_id: str) -> None:
            """Fill the info panel (name, types, description) + a detailed tooltip
            (description + single-op pseudocode) for the given op id."""
            op = REGISTRY.get(op_id)
            info_name.setText(op.label if op is not None else op_id)   # display label, not id
            if op is None:
                info_types.setText(""); info_desc.setText(""); return
            info_types.setText(f"Input: {op.in_label or '?'}\nOutput: {op.out_label or '?'}")
            desc = codegen.op_description(op)
            info_desc.setText(desc)
            tip = desc + "\n\n" + codegen.op_pseudocode(op)
            for _w in (info_name, info_types, info_desc, info_group):
                _w.setToolTip(tip)
        
        # Parameter controls Ã¢â‚¬â€ auto-generated from the selected op's schema.
        param_group = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QVBoxLayout()
        param_group.setLayout(param_layout)
        param_group.setVisible(False)  # Hidden by default
        self.param_panel = ParameterPanel()
        param_layout.addWidget(self.param_panel)


        sidebar.layout().addWidget(title)
        sidebar.layout().addLayout(toolbar)
        sidebar.layout().addWidget(search)
        sidebar.layout().addWidget(tree, 2)

        bottom = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_layout.addWidget(param_group)
        bottom_layout.addWidget(info_group)
        bottom_layout.addStretch(1)
        sidebar.layout().addWidget(bottom, 1)

        # Main drop/view pane
        self.drop_widget = ImageDropWidget()

        # Live inspector for the currently selected node (right side).
        self.inspector_pane = InspectorPane()

        splitter.addWidget(sidebar)
        splitter.addWidget(self.drop_widget)
        splitter.addWidget(self.inspector_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 900, 380])   # give the inspector pane a bit more width

        container = QtWidgets.QWidget()
        container.setLayout(QtWidgets.QHBoxLayout())
        container.layout().setContentsMargins(8, 8, 8, 8)
        container.layout().addWidget(splitter)
        self.setCentralWidget(container)

        open_btn.clicked.connect(self.drop_widget.browse_for_image)
        open_imgs_btn.clicked.connect(self.drop_widget.browse_for_images)
        save_btn.clicked.connect(self.save_pipeline)
        load_btn.clicked.connect(self.load_pipeline)
        self.drop_widget.imageLoaded.connect(self.on_image_loaded)
        
        # Connect double-click signal to open image viewer
        self.drop_widget.view.nodeDoubleClicked.connect(self.open_image_viewer)

        # Keep the live inspector pane in sync with node result changes and with
        # the previewed batch element (which the canvas wheel can also change).
        self.drop_widget.view.controller.signals.nodeChanged.connect(self._on_pane_node_changed)
        self.drop_widget.view.controller.signals.previewIndexChanged.connect(
            lambda _i: self.inspector_pane.refresh())


        def on_tree_selection_changed() -> None:
            items = tree.selectedItems()
            if not items:
                info_name.setText("Select a function")
                info_types.setText("")
                return
            item = items[0]
            meta = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(meta, dict):
                set_func_info(meta.get('name', ''))
            else:
                info_name.setText("Select a function")
                info_types.setText(""); info_desc.setText("")

        tree.itemSelectionChanged.connect(on_tree_selection_changed)

        def on_tree_item_double_clicked(item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
            meta = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(meta, dict) and 'name' in meta:
                # Add a function node to the scene with the display label
                self.drop_widget.add_function_node(item.text(0), None, meta)
                # Also update info box to reflect the selected function
                set_func_info(meta.get('name', ''))

        tree.itemDoubleClicked.connect(on_tree_item_double_clicked)

        # Update info panel when selecting a function node in the scene
        def on_scene_selection_changed() -> None:
            try:
                selected = self.drop_widget.view._scene.selectedItems()
                if not selected:
                    self.param_panel.clear()
                    param_group.setVisible(False)
                    self.inspector_pane.set_node(None)
                    return
                sel = selected[0]
                if isinstance(sel, FunctionNode):
                    self.inspector_pane.set_node(sel)
                    meta = getattr(sel, "_meta", None)
                    if isinstance(meta, dict):
                        set_func_info(meta.get('name', ''))
                        # Auto-build parameter controls from the op's schema.
                        self.param_panel.set_node(sel)
                        param_group.setVisible(self.param_panel.has_controls())
                elif isinstance(sel, ImageNode):
                    self.inspector_pane.set_node(sel)
                    meta = getattr(sel, "_meta", None)
                    if isinstance(meta, dict):
                        info_name.setText("Image")
                        info_desc.setText("")
                        ch = meta.get('channels', '?')
                        info_types.setText(
                            f"Size: {meta.get('w','?')}ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â{meta.get('h','?')}\n"
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

    def _on_pane_node_changed(self, qt_node) -> None:
        """Refresh the live inspector pane when its node recomputes."""
        if qt_node is self.inspector_pane._node:
            self.inspector_pane.refresh()

    @staticmethod
    def _pipeline_dir() -> str:
        """Default directory for the pipeline dialogs (created if missing)."""
        d = Path("test") / "pipelines"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return str(d)

    def save_pipeline(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Pipeline", self._pipeline_dir(), "Pipeline (*.json);;All Files (*)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.drop_widget.to_dict(), f)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def load_pipeline(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Pipeline", self._pipeline_dir(), "Pipeline (*.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.drop_widget.load_dict(data)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
