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
        open_btn = QtWidgets.QPushButton("Open ImageÃ¢â‚¬Â¦")
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
        
        # Parameter controls (initially hidden)
        param_group = QtWidgets.QGroupBox("Parameters")
        param_layout = QtWidgets.QVBoxLayout()
        param_group.setLayout(param_layout)
        param_group.setVisible(False)  # Hidden by default
        
        # Parameter widgets (will be populated dynamically)
        self.param_widgets = {}

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
                    param_group.setVisible(False)
                    return
                sel = selected[0]
                if isinstance(sel, FunctionNode):
                    meta = getattr(sel, "_meta", None)
                    if isinstance(meta, dict):
                        info_name.setText(f"{meta.get('name','')}()")
                        info_types.setText(f"Input: {meta.get('in','?')}\nOutput: {meta.get('out','?')}")
                        # Show parameter controls for this function
                        self._setup_parameter_controls(sel, param_group, param_layout)
                        param_group.setVisible(True)
                elif isinstance(sel, ImageNode):
                    meta = getattr(sel, "_meta", None)
                    if isinstance(meta, dict):
                        info_name.setText("Image")
                        ch = meta.get('channels', '?')
                        info_types.setText(
                            f"Size: {meta.get('w','?')}Ãƒâ€”{meta.get('h','?')}\n"
                            f"Channels: {ch}\n"
                            f"Type: {meta.get('type','?')}"
                        )
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
    
    def _setup_parameter_controls(self, function_item: FunctionNode, param_group: QtWidgets.QGroupBox, param_layout: QtWidgets.QVBoxLayout) -> None:
        """Setup parameter controls for the selected function node."""
        # Clear existing widgets properly
        while param_layout.count():
            child = param_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        func_name = function_item._meta.get("name", "")
        params = function_item.get_parameters()
        
        if not params:
            return
        
        if func_name == "blur":
            # Kernel size slider
            kernel_label = QtWidgets.QLabel("Kernel Size:")
            kernel_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            kernel_slider.setRange(1, 101)  # Odd numbers only
            kernel_slider.setValue(params.get("kernel_size", 15))
            kernel_slider.setSingleStep(2)  # Keep odd numbers
            kernel_value = QtWidgets.QLabel(str(params.get("kernel_size", 15)))
            
            def on_kernel_changed(value):
                # Ensure odd number
                odd_value = value if value % 2 == 1 else value + 1
                kernel_slider.setValue(odd_value)
                kernel_value.setText(str(odd_value))
                function_item.set_parameter("kernel_size", odd_value, preview_mode=True)
            
            def on_kernel_released():
                # Final execution when slider is released
                function_item.set_parameter("kernel_size", kernel_slider.value(), preview_mode=False)
            
            kernel_slider.valueChanged.connect(on_kernel_changed)
            kernel_slider.sliderReleased.connect(on_kernel_released)
            
            param_layout.addWidget(kernel_label)
            param_layout.addWidget(kernel_slider)
            param_layout.addWidget(kernel_value)
            
        elif func_name == "threshold":
            # Threshold value slider
            thresh_label = QtWidgets.QLabel("Threshold Value:")
            thresh_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            thresh_slider.setRange(0, 255)
            thresh_slider.setValue(params.get("threshold_value", 127))
            thresh_value = QtWidgets.QLabel(str(params.get("threshold_value", 127)))
            
            def on_thresh_changed(value):
                thresh_value.setText(str(value))
                function_item.set_parameter("threshold_value", value, preview_mode=True)
            
            def on_thresh_released():
                # Final execution when slider is released
                function_item.set_parameter("threshold_value", thresh_slider.value(), preview_mode=False)
            
            thresh_slider.valueChanged.connect(on_thresh_changed)
            thresh_slider.sliderReleased.connect(on_thresh_released)
            
            # Threshold type dropdown
            thresh_type_label = QtWidgets.QLabel("Threshold Type:")
            thresh_type_combo = QtWidgets.QComboBox()
            thresh_type_combo.addItem("Binary", cv2.THRESH_BINARY)
            thresh_type_combo.addItem("Binary Inv", cv2.THRESH_BINARY_INV)
            thresh_type_combo.addItem("Trunc", cv2.THRESH_TRUNC)
            thresh_type_combo.addItem("To Zero", cv2.THRESH_TOZERO)
            thresh_type_combo.addItem("To Zero Inv", cv2.THRESH_TOZERO_INV)
            
            # Set current value
            current_thresh_type = params.get("threshold_type", cv2.THRESH_BINARY)
            thresh_type_mapping = {
                cv2.THRESH_BINARY: 0,
                cv2.THRESH_BINARY_INV: 1,
                cv2.THRESH_TRUNC: 2,
                cv2.THRESH_TOZERO: 3,
                cv2.THRESH_TOZERO_INV: 4
            }
            thresh_type_combo.setCurrentIndex(thresh_type_mapping.get(current_thresh_type, 0))
            
            def on_thresh_type_changed():
                function_item.set_parameter("threshold_type", thresh_type_combo.currentData(), preview_mode=False)
            
            thresh_type_combo.currentIndexChanged.connect(on_thresh_type_changed)
            
            param_layout.addWidget(thresh_label)
            param_layout.addWidget(thresh_slider)
            param_layout.addWidget(thresh_value)
            param_layout.addWidget(thresh_type_label)
            param_layout.addWidget(thresh_type_combo)
            
        elif func_name == "adaptive_threshold":
            # Max value slider
            max_value_label = QtWidgets.QLabel("Max Value:")
            max_value_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            max_value_slider.setRange(1, 255)
            max_value_slider.setValue(params.get("max_value", 255))
            max_value_value = QtWidgets.QLabel(str(params.get("max_value", 255)))
            
            def on_max_value_changed(value):
                max_value_value.setText(str(value))
                function_item.set_parameter("max_value", value, preview_mode=True)
            
            def on_max_value_released():
                function_item.set_parameter("max_value", max_value_slider.value(), preview_mode=False)
            
            max_value_slider.valueChanged.connect(on_max_value_changed)
            max_value_slider.sliderReleased.connect(on_max_value_released)
            
            # Adaptive method dropdown
            method_label = QtWidgets.QLabel("Adaptive Method:")
            method_combo = QtWidgets.QComboBox()
            method_combo.addItem("Mean C", cv2.ADAPTIVE_THRESH_MEAN_C)
            method_combo.addItem("Gaussian C", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
                        
            # Set current value
            current_method = params.get("adaptive_method", cv2.ADAPTIVE_THRESH_MEAN_C)
            if current_method == cv2.ADAPTIVE_THRESH_GAUSSIAN_C:
                method_combo.setCurrentIndex(1)
            else:
                method_combo.setCurrentIndex(0)
            
            def on_method_changed():
                function_item.set_parameter("adaptive_method", method_combo.currentData(), preview_mode=False)
            
            method_combo.currentIndexChanged.connect(on_method_changed)
            
            # Threshold type dropdown
            thresh_type_label = QtWidgets.QLabel("Threshold Type:")
            thresh_type_combo = QtWidgets.QComboBox()
            thresh_type_combo.addItem("Binary", cv2.THRESH_BINARY)
            thresh_type_combo.addItem("Binary Inv", cv2.THRESH_BINARY_INV)
            
            # Set current value
            current_thresh_type = params.get("threshold_type", cv2.THRESH_BINARY)
            thresh_type_mapping = {
                cv2.THRESH_BINARY: 0,
                cv2.THRESH_BINARY_INV: 1
            }
            thresh_type_combo.setCurrentIndex(thresh_type_mapping.get(current_thresh_type, 0))
            
            def on_thresh_type_changed():
                function_item.set_parameter("threshold_type", thresh_type_combo.currentData(), preview_mode=False)
            
            thresh_type_combo.currentIndexChanged.connect(on_thresh_type_changed)
            
            # Block size slider
            block_size_label = QtWidgets.QLabel("Block Size:")
            block_size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            block_size_slider.setRange(3, 51)  # Must be odd numbers
            block_size_slider.setValue(params.get("block_size", 11))
            block_size_slider.setSingleStep(2)  # Keep odd numbers
            block_size_value = QtWidgets.QLabel(str(params.get("block_size", 11)))
            
            def on_block_size_changed(value):
                # Ensure odd number
                odd_value = value if value % 2 == 1 else value + 1
                block_size_slider.setValue(odd_value)
                block_size_value.setText(str(odd_value))
                function_item.set_parameter("block_size", odd_value, preview_mode=True)
            
            def on_block_size_released():
                function_item.set_parameter("block_size", block_size_slider.value(), preview_mode=False)
            
            block_size_slider.valueChanged.connect(on_block_size_changed)
            block_size_slider.sliderReleased.connect(on_block_size_released)
            
            # C value slider
            c_label = QtWidgets.QLabel("C Value:")
            c_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            c_slider.setRange(-10, 10)
            c_slider.setValue(params.get("c", 2))
            c_value = QtWidgets.QLabel(str(params.get("c", 2)))
            
            def on_c_changed(value):
                c_value.setText(str(value))
                function_item.set_parameter("c", value, preview_mode=True)
            
            def on_c_released():
                function_item.set_parameter("c", c_slider.value(), preview_mode=False)
            
            c_slider.valueChanged.connect(on_c_changed)
            c_slider.sliderReleased.connect(on_c_released)
            
            param_layout.addWidget(max_value_label)
            param_layout.addWidget(max_value_slider)
            param_layout.addWidget(max_value_value)
            param_layout.addWidget(method_label)
            param_layout.addWidget(method_combo)
            param_layout.addWidget(thresh_type_label)
            param_layout.addWidget(thresh_type_combo)
            param_layout.addWidget(block_size_label)
            param_layout.addWidget(block_size_slider)
            param_layout.addWidget(block_size_value)
            param_layout.addWidget(c_label)
            param_layout.addWidget(c_slider)
            param_layout.addWidget(c_value)
            
        elif func_name == "save_to_file":
            # Custom filename checkbox
            custom_checkbox = QtWidgets.QCheckBox("Use custom filename")
            custom_checkbox.setChecked(params.get("use_custom", False))
            
            # Filename input
            filename_input = QtWidgets.QLineEdit()
            filename_input.setText(params.get("filename", ""))
            filename_input.setPlaceholderText("Enter custom filename (optional)")
            filename_input.setEnabled(params.get("use_custom", False))
            
            # Browse button
            browse_btn = QtWidgets.QPushButton("Browse...")
            browse_btn.setEnabled(params.get("use_custom", False))
            
            def on_custom_toggled(checked):
                filename_input.setEnabled(checked)
                browse_btn.setEnabled(checked)
                function_item.set_parameter("use_custom", checked)
            
            def on_filename_changed():
                function_item.set_parameter("filename", filename_input.text())
            
            def on_browse_clicked():
                from PyQt6.QtWidgets import QFileDialog
                filename, _ = QFileDialog.getSaveFileName(
                    None, 
                    "Save Image As", 
                    "./output/", 
                    "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
                )
                if filename:
                    filename_input.setText(filename)
                    function_item.set_parameter("filename", filename)
            
            custom_checkbox.toggled.connect(on_custom_toggled)
            filename_input.textChanged.connect(on_filename_changed)
            browse_btn.clicked.connect(on_browse_clicked)
            
            param_layout.addWidget(custom_checkbox)
            param_layout.addWidget(QtWidgets.QLabel("Filename:"))
            param_layout.addWidget(filename_input)
            param_layout.addWidget(browse_btn)
            
        elif func_name == "sum":
            # Alpha parameter slider
            alpha_label = QtWidgets.QLabel("Alpha (Weight):")
            alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            alpha_slider.setRange(0, 100)  # 0-100 for percentage
            alpha_slider.setValue(int(params.get("alpha", 0.5) * 100))
            alpha_value = QtWidgets.QLabel(f"{params.get('alpha', 0.5):.2f}")
            
            def on_alpha_changed(value):
                alpha = value / 100.0
                alpha_value.setText(f"{alpha:.2f}")
                function_item.set_parameter("alpha", alpha, preview_mode=True)
            
            def on_alpha_released():
                function_item.set_parameter("alpha", alpha_slider.value() / 100.0, preview_mode=False)
            
            alpha_slider.valueChanged.connect(on_alpha_changed)
            alpha_slider.sliderReleased.connect(on_alpha_released)
            
            param_layout.addWidget(alpha_label)
            param_layout.addWidget(alpha_slider)
            param_layout.addWidget(alpha_value)
            
        elif func_name == "mser":
            # Delta parameter
            delta_label = QtWidgets.QLabel("Delta:")
            delta_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            delta_slider.setRange(1, 20)
            delta_slider.setValue(params.get("delta", 5))
            delta_value = QtWidgets.QLabel(str(params.get("delta", 5)))
            
            def on_delta_changed(value):
                delta_value.setText(str(value))
                function_item.set_parameter("delta", value, preview_mode=True)
            
            def on_delta_released():
                function_item.set_parameter("delta", delta_slider.value(), preview_mode=False)
            
            delta_slider.valueChanged.connect(on_delta_changed)
            delta_slider.sliderReleased.connect(on_delta_released)
            
            # Min Area parameter
            min_area_label = QtWidgets.QLabel("Min Area:")
            min_area_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            min_area_slider.setRange(10, 1000)
            min_area_slider.setValue(params.get("min_area", 60))
            min_area_value = QtWidgets.QLabel(str(params.get("min_area", 60)))
            
            def on_min_area_changed(value):
                min_area_value.setText(str(value))
                function_item.set_parameter("min_area", value, preview_mode=True)
            
            def on_min_area_released():
                function_item.set_parameter("min_area", min_area_slider.value(), preview_mode=False)
            
            min_area_slider.valueChanged.connect(on_min_area_changed)
            min_area_slider.sliderReleased.connect(on_min_area_released)
            
            # Max Area parameter
            max_area_label = QtWidgets.QLabel("Max Area:")
            max_area_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            max_area_slider.setRange(1000, 50000)
            max_area_slider.setValue(params.get("max_area", 14400))
            max_area_value = QtWidgets.QLabel(str(params.get("max_area", 14400)))
            
            def on_max_area_changed(value):
                max_area_value.setText(str(value))
                function_item.set_parameter("max_area", value, preview_mode=True)
            
            def on_max_area_released():
                function_item.set_parameter("max_area", max_area_slider.value(), preview_mode=False)
            
            max_area_slider.valueChanged.connect(on_max_area_changed)
            max_area_slider.sliderReleased.connect(on_max_area_released)
            
            # Max Variation parameter
            max_var_label = QtWidgets.QLabel("Max Variation:")
            max_var_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            max_var_slider.setRange(0, 100)  # 0.0 to 1.0 as percentage
            max_var_slider.setValue(int(params.get("max_variation", 0.25) * 100))
            max_var_value = QtWidgets.QLabel(f"{params.get('max_variation', 0.25):.2f}")
            
            def on_max_var_changed(value):
                max_var = value / 100.0
                max_var_value.setText(f"{max_var:.2f}")
                function_item.set_parameter("max_variation", max_var, preview_mode=True)
            
            def on_max_var_released():
                function_item.set_parameter("max_variation", max_var_slider.value() / 100.0, preview_mode=False)
            
            max_var_slider.valueChanged.connect(on_max_var_changed)
            max_var_slider.sliderReleased.connect(on_max_var_released)
            
            # Min Diversity parameter
            min_div_label = QtWidgets.QLabel("Min Diversity:")
            min_div_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            min_div_slider.setRange(0, 100)  # 0.0 to 1.0 as percentage
            min_div_slider.setValue(int(params.get("min_diversity", 0.2) * 100))
            min_div_value = QtWidgets.QLabel(f"{params.get('min_diversity', 0.2):.2f}")
            
            def on_min_div_changed(value):
                min_div = value / 100.0
                min_div_value.setText(f"{min_div:.2f}")
                function_item.set_parameter("min_diversity", min_div, preview_mode=True)
            
            def on_min_div_released():
                function_item.set_parameter("min_diversity", min_div_slider.value() / 100.0, preview_mode=False)
            
            min_div_slider.valueChanged.connect(on_min_div_changed)
            min_div_slider.sliderReleased.connect(on_min_div_released)
            
            param_layout.addWidget(delta_label)
            param_layout.addWidget(delta_slider)
            param_layout.addWidget(delta_value)
            param_layout.addWidget(min_area_label)
            param_layout.addWidget(min_area_slider)
            param_layout.addWidget(min_area_value)
            param_layout.addWidget(max_area_label)
            param_layout.addWidget(max_area_slider)
            param_layout.addWidget(max_area_value)
            param_layout.addWidget(max_var_label)
            param_layout.addWidget(max_var_slider)
            param_layout.addWidget(max_var_value)
            param_layout.addWidget(min_div_label)
            param_layout.addWidget(min_div_slider)
            param_layout.addWidget(min_div_value)


