"""The node canvas: the graphics view and its drop-target wrapper (frontend)."""
from pathlib import Path
from typing import Optional

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from core.operations import by_label
from core import persistence
from core.batch import Batch
from ui.controller import GraphController
from ui.nodes import Node, ImageNode, FunctionNode, SaveToFileNode, ExportCodeNode
from ui.arrow import ArrowItem

# Default node icon size (px). Owned here since the canvas manages icons.
DEFAULT_ICON_SIZE = 90

class GraphicsImageView(QtWidgets.QGraphicsView):
    fileDropped = QtCore.pyqtSignal(Path, QtCore.QPointF)
    arrowCreated = QtCore.pyqtSignal(QtWidgets.QGraphicsItem, QtWidgets.QGraphicsItem)
    nodeDoubleClicked = QtCore.pyqtSignal(Node)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        # The backend graph + evaluator for everything on this canvas.
        self.controller = GraphController()
        # Highlight the selected node (yellow) + its whole data flow (green).
        self._scene.selectionChanged.connect(self._update_flow_highlight)
        # Scene will follow the viewport size to fill the right side
        # A roomy canvas (~2x the viewport, growing to fit the nodes), scrollable
        # and zoomable (Ctrl+wheel) for large pipelines.
        self._zoom_level = 1.0
        self._update_scene_rect()
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Accept drops directly on the view
        self.setAcceptDrops(True)
        # Grid parameters (hover overlay disabled)
        self._grid_size = 12
        self._thumb_size = DEFAULT_ICON_SIZE
        # Arrow creation state
        self._right_dragging = False
        self._start_item: Optional[Node] = None
        self._hover_item: Optional[Node] = None
        self._destination_item: Optional[Node] = None

    def _update_flow_highlight(self) -> None:
        """On a single-node selection, tint that node yellow, every predecessor
        and successor green, and the edges between flow nodes green. Cleared when
        nothing (or more than one node) is selected."""
        nodes = [it for it in self._scene.selectedItems() if isinstance(it, Node)]
        selected_gn = nodes[0].gnode if len(nodes) == 1 and getattr(nodes[0], "gnode", None) else None
        flow_ids = set()
        if selected_gn is not None:
            model = self.controller.model
            flow_ids = {selected_gn.id} | model.ancestors(selected_gn) | model.descendants(selected_gn)
        for it in self._scene.items():
            if isinstance(it, Node):
                gn = getattr(it, "gnode", None)
                if gn is not None and selected_gn is not None and gn.id == selected_gn.id:
                    it.set_flow_role("selected")
                elif gn is not None and gn.id in flow_ids:
                    it.set_flow_role("flow")
                else:
                    it.set_flow_role(None)
            elif isinstance(it, ArrowItem):
                a, b = getattr(it.a, "gnode", None), getattr(it.b, "gnode", None)
                it.set_flow_highlight(a is not None and b is not None
                                      and a.id in flow_ids and b.id in flow_ids)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._update_scene_rect()
        super().resizeEvent(event)

    def _update_scene_rect(self) -> None:
        """Fixed coordinate origin: the scene's top-left is pinned at (0, 0) and it
        only ever grows right/down to enclose the nodes — it never shifts the
        origin. So every node's (x, y) is a stable absolute position: zoom and
        scroll move all nodes and the grid together, scaling a node keeps its
        top-left fixed, and save/load round-trips positions exactly (the layout is
        never squashed into a moved or too-small rect)."""
        vw = max(self.viewport().width(), 400)
        vh = max(self.viewport().height(), 400)
        w, h = vw * 2.0, vh * 2.0
        items = self._scene.itemsBoundingRect()
        if not items.isEmpty():
            w = max(w, items.right() + 200)
            h = max(h, items.bottom() + 200)
        self._scene.setSceneRect(0, 0, w, h)

    def _zoom(self, factor: float) -> None:
        target = max(0.3, min(3.0, self._zoom_level * factor))
        factor = target / self._zoom_level
        if abs(factor - 1.0) < 1e-3:
            return
        self._zoom_level = target
        self.scale(factor, factor)

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        # Fill background gray
        painter.save()
        try:
            painter.fillRect(rect, QtGui.QColor(240, 240, 240))
            # Draw grid lines every 12 px
            grid = 12
            left = int(rect.left()) - (int(rect.left()) % grid)
            top = int(rect.top()) - (int(rect.top()) % grid)
            pen_minor = QtGui.QPen(QtGui.QColor(220, 220, 220))
            pen_minor.setWidth(1)
            pen_major = QtGui.QPen(QtGui.QColor(200, 200, 200))
            pen_major.setWidth(1)

            # Vertical lines
            x = left
            while x < rect.right():
                pen = pen_major if (x % (grid * 5) == 0) else pen_minor
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
                x += grid

            # Horizontal lines
            y = top
            while y < rect.bottom():
                pen = pen_major if (y % (grid * 5) == 0) else pen_minor
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
                y += grid
        finally:
            painter.restore()

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        # Hover highlight removed
        super().drawForeground(painter, rect)

    # Hover update removed

    def set_thumb_size(self, size: int) -> None:
        self._thumb_size = int(size)
        # Redraw view to reflect size-dependent overlays (if any)
        self.viewport().update()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        view_pos = event.position().toPoint()
        scene_pos = self.mapToScene(view_pos)
        for url in event.mimeData().urls():
            local_path = Path(url.toLocalFile())
            if local_path.exists():
                self.fileDropped.emit(local_path, scene_pos)
                break
        event.acceptProposedAction()
        # No hover to clear

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._right_dragging:
            self._update_hover_icon(self.mapToScene(event.position().toPoint()))
            self._update_destination_highlight(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self._right_dragging = True
            pos = self.mapToScene(event.position().toPoint())
            self._start_item = self._nearest_icon(pos)
            self._set_highlight(self._start_item, True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._right_dragging and event.button() == QtCore.Qt.MouseButton.RightButton:
            self._right_dragging = False
            end_item = self._hover_item if (self._hover_item is not None and self._hover_item is not self._start_item) else None
            self._set_highlight(self._hover_item, False)
            self._set_destination_highlight(self._destination_item, False)
            if self._start_item is not None:
                self._set_highlight(self._start_item, False)
            if self._start_item is not None and end_item is not None:
                self._create_arrow_between(self._start_item, end_item)
            self._start_item = None
            self._hover_item = None
            self._destination_item = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle double-click events to open image viewer."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Find the item at the click position
            scene_pos = self.mapToScene(event.position().toPoint())
            item = self._nearest_icon(scene_pos)
            
            if item is not None and isinstance(item, Node):
                # Check if the item has an output image
                output_image = item.get_output_image()
                if output_image is not None:
                    self.nodeDoubleClicked.emit(item)
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self._delete_selected()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_S:
            # Swap the two inputs of a selected binary op (e.g. Diff A<->B).
            funcs = [it for it in self._scene.selectedItems() if isinstance(it, FunctionNode)]
            if len(funcs) == 1:
                self.controller.swap_inputs(funcs[0])
                event.accept()
                return
        super().keyPressEvent(event)

    def _delete_selected(self) -> None:
        for item in list(self._scene.selectedItems()):
            if isinstance(item, ArrowItem):
                self._delete_arrow(item)
            elif isinstance(item, Node):
                self._delete_node(item)

    def _detach_arrow(self, arrow: ArrowItem) -> None:
        arrow.a._unregister_arrow(arrow)
        arrow.b._unregister_arrow(arrow)
        if arrow.scene() is not None:
            self._scene.removeItem(arrow)

    def _delete_arrow(self, arrow: ArrowItem) -> None:
        a, b = arrow.a, arrow.b            # a = source, b = target (see _create_arrow_between)
        self._detach_arrow(arrow)
        self.controller.delete_edge(a, b)

    def _delete_node(self, node: Node) -> None:
        for arrow in list(node._arrows):
            self._detach_arrow(arrow)
        self.controller.unregister(node)   # drops the backend node + its edges, then recomputes
        if node.scene() is not None:
            self._scene.removeItem(node)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        # Ctrl+wheel zooms the canvas.
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            self._zoom(1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15)
            event.accept()
            return
        # Wheel directly over a batch node scrolls its frames; otherwise it
        # scrolls the (now larger) canvas as usual.
        item = self._scene.itemAt(self.mapToScene(event.position().toPoint()), self.transform())
        node = item if isinstance(item, Node) else None
        step = 1 if event.angleDelta().y() < 0 else -1
        if self._scroll_batch(node, step):
            event.accept()
            return
        super().wheelEvent(event)

    def _batch_len(self, node) -> int:
        gn = getattr(node, "gnode", None)
        if gn is None:
            return 1
        val = gn.source_image if gn.is_source else getattr(gn, "output", None)
        return len(val) if isinstance(val, Batch) else 1

    def _scroll_batch(self, node, step: int) -> bool:
        if node is None:
            return False
        n = self._batch_len(node)
        if n <= 1:
            return False
        self.controller.set_preview_index(
            max(0, min(self.controller.preview_index + step, n - 1)))
        return True

    def _nearest_icon(self, scene_pos: QtCore.QPointF) -> Optional[Node]:
        nearest: Optional[Node] = None
        best_dist2 = float('inf')
        for item in self._scene.items():
            if isinstance(item, Node):
                center = item.sceneBoundingRect().center()
                dx = center.x() - scene_pos.x()
                dy = center.y() - scene_pos.y()
                d2 = dx*dx + dy*dy
                if d2 < best_dist2:
                    best_dist2 = d2
                    nearest = item
        return nearest

    def _update_hover_icon(self, scene_pos: QtCore.QPointF) -> None:
        candidate = self._nearest_icon(scene_pos)
        if candidate is self._start_item:
            candidate = None
        if candidate is self._hover_item:
            return
        # Update highlight states
        self._set_highlight(self._hover_item, False)
        self._hover_item = candidate
        self._set_highlight(self._hover_item, True)

    def _set_highlight(self, item: Optional[Node], highlighted: bool) -> None:
        if item is not None:
            item.set_highlighted(highlighted)

    def _update_destination_highlight(self, scene_pos: QtCore.QPointF) -> None:
        """Update destination highlighting based on cursor position."""
        candidate = self._nearest_icon(scene_pos)
        
        # Don't highlight the start item as destination
        if candidate is self._start_item:
            candidate = None
        
        # Update destination highlighting
        if candidate != self._destination_item:
            # Clear previous destination highlight
            self._set_destination_highlight(self._destination_item, False)
            
            # Set new destination highlight
            self._destination_item = candidate
            if candidate is not None:
                # Check connection type and set appropriate highlighting
                connection_type = self._get_connection_type(self._start_item, candidate)
                if connection_type == 'valid':
                    self._set_destination_highlight(candidate, True, True)
                elif connection_type == 'implicit_conversion':
                    self._set_destination_highlight(candidate, True, 'implicit_conversion')
                else:  # invalid
                    self._set_destination_highlight(candidate, True, False)
    
    def _set_destination_highlight(self, item: Optional[Node], highlighted: bool, is_valid: bool = True) -> None:
        """Set destination highlighting with color based on validity."""
        if item is not None:
            item.set_destination_highlighted(highlighted, is_valid)
    
    def _is_valid_connection(self, source: Optional[Node], target: Optional[Node]) -> bool:
        """Check if a connection between source and target would be valid."""
        if source is None or target is None:
            return False
        
        # Check if source is SaveToFile (not allowed as source)
        if isinstance(source, SaveToFileNode):
            return False

        # Already connected -> dragging again disconnects (still a valid action).
        if self.controller.is_connected(source, target):
            return True
        # Accept either a normal connection or a rewire of a full single-input node.
        if target.can_accept_input(source):
            return True
        return isinstance(target, FunctionNode) and self.controller.can_rewire(source, target)

    def _get_connection_type(self, source: Optional[Node], target: Optional[Node]) -> str:
        """Get the type of connection: 'valid', 'invalid', or 'implicit_conversion'."""
        if source is None or target is None:
            return 'invalid'

        # Check if source is SaveToFile (not allowed as source)
        if isinstance(source, SaveToFileNode):
            return 'invalid'

        # Already connected -> dragging again disconnects.
        if self.controller.is_connected(source, target):
            return 'valid'

        # Check if target can accept input from source
        if not target.can_accept_input(source):
            if isinstance(target, FunctionNode) and self.controller.can_rewire(source, target):
                return 'valid'   # full single-input target -> rewire
            return 'invalid'

        # Check for implicit conversion cases
        if self._needs_implicit_conversion(source, target):
            return 'implicit_conversion'

        return 'valid'
    
    def _needs_implicit_conversion(self, source: Optional[Node], target: Optional[Node]) -> bool:
        """Check if an implicit conversion will be needed for this connection."""
        if source is None or target is None:
            return False
        
        # Get the output image from source to check its type
        source_image = source.get_output_image()
        import numpy as np
        if not isinstance(source_image, np.ndarray):
            return False  # non-image output (e.g. clusters) — no implicit conversion

        # Check if source outputs BGR and target expects grayscale
        if len(source_image.shape) == 3 and source_image.shape[2] == 3:  # BGR image
            # Check if target is a grayscale-only function
            if isinstance(target, FunctionNode) and target.op.id in ("threshold", "adaptive_threshold"):
                return True
        
        return False

    def _create_arrow_between(self, a: Node, b: Node) -> None:
        # Allow connections between any nodes that can provide and accept data
        source_node = None
        target_node = None
        
        # Determine source and target based on node types
        if isinstance(a, ImageNode) and isinstance(b, FunctionNode):
            source_node, target_node = a, b
        elif isinstance(a, FunctionNode) and isinstance(b, ImageNode):
            source_node, target_node = b, a
        elif isinstance(a, FunctionNode) and isinstance(b, FunctionNode):
            # Allow function-to-function connections, but not SaveToFile as source
            if isinstance(a, SaveToFileNode):
                return  # SaveToFile cannot be source for other functions
            source_node, target_node = a, b
        else:
            return  # Invalid arrow combination
        
        if self.controller.is_connected(source_node, target_node):
            # Dragging onto an already-connected target toggles the link off.
            self._disconnect(source_node, target_node)
        elif target_node.can_accept_input(source_node):
            # Create the arrow and register the connection.
            self._scene.addItem(ArrowItem(source_node, target_node))
            target_node.add_input_connection(source_node)
        elif self.controller.can_rewire(source_node, target_node):
            # Target's single input is full -> re-point it at the new source.
            self._rewire(source_node, target_node)

    def _disconnect(self, source_node: Node, target_node: Node) -> None:
        for arrow in list(target_node._arrows):
            if source_node in (arrow.a, arrow.b) and target_node in (arrow.a, arrow.b):
                self._detach_arrow(arrow)
        self.controller.delete_edge(source_node, target_node)

    def _rewire(self, source_node: Node, target_node: Node) -> None:
        """Replace a full single-input node's connection with one from source_node."""
        old_edges = self.controller.model.incoming(target_node.gnode)
        old_src_qt = (self.controller._qt_by_gid.get(old_edges[0].src.id)
                      if old_edges else None)
        if old_src_qt is not None:
            for arrow in list(target_node._arrows):
                if old_src_qt in (arrow.a, arrow.b) and target_node in (arrow.a, arrow.b):
                    self._detach_arrow(arrow)
                    break
        self.controller.replace_input(source_node, target_node)
        self._scene.addItem(ArrowItem(source_node, target_node))

    






class ImageDropWidget(QtWidgets.QWidget):
    imageLoaded = QtCore.pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setContentsMargins(8, 8, 8, 8)
        self.layout().setSpacing(8)

        self.instruction = QtWidgets.QLabel("Drop an image here or click to browse\n(icons snap to 12x12 grid)")
        self.instruction.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.instruction.setStyleSheet("color: #666;")

        self.view = GraphicsImageView()
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.icon_size = DEFAULT_ICON_SIZE  # central icon size constant (px)
        self.view.set_thumb_size(self.icon_size)

        # Icon-size control at the top-left of the pipeline pane.
        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self._size_label = QtWidgets.QLabel(f"Icon size: {self.icon_size} px")
        self._size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._size_slider.setRange(24, 256)
        self._size_slider.setValue(self.icon_size)
        self._size_slider.setFixedWidth(140)
        self._size_slider.valueChanged.connect(self._on_icon_size_changed)
        controls.addWidget(self._size_label)
        controls.addWidget(self._size_slider)
        controls.addStretch()
        self.layout().addLayout(controls)

        border = QtWidgets.QFrame()
        border.setFrameShape(QtWidgets.QFrame.Shape.Box)
        border.setLineWidth(1)
        border.setLayout(QtWidgets.QVBoxLayout())
        border.layout().setContentsMargins(16, 16, 16, 16)
        border.layout().addWidget(self.instruction)
        border.layout().addWidget(self.view)

        self.layout().addWidget(border)

        # Connect view's drop signal to add icons with snapping
        self.view.fileDropped.connect(self.on_file_dropped)

    def _on_icon_size_changed(self, value: int) -> None:
        self.icon_size = int(value)
        self._size_label.setText(f"Icon size: {value} px")
        self.view.set_thumb_size(self.icon_size)
        self.resize_all_icons(self.icon_size)
        self.view._update_scene_rect()   # grow the scene if bigger icons overflow it

    def add_function_node(self, label: str, scene_pos: Optional[QtCore.QPointF] = None, meta: Optional[dict] = None):
        # Look the operation up in the registry and build the right node.
        op = by_label.get(label)
        if op is None:
            return None
        if op.id == "save_to_file":
            item = SaveToFileNode(icon_size=self.icon_size, grid_size=12)
        elif op.id == "export_code":
            item = ExportCodeNode(icon_size=self.icon_size, grid_size=12)
        else:
            item = FunctionNode(op, icon_size=self.icon_size, grid_size=12)

        self.view.controller.register_op(item)
        self.view._scene.addItem(item)
        self.view._update_scene_rect()
        if scene_pos is None:
            # Drop into the middle of what's currently visible (not the middle of
            # the much larger scene, which may be scrolled off-screen).
            center = self.view.mapToScene(self.view.viewport().rect().center())
            half = self.icon_size / 2
            scene_pos = QtCore.QPointF(center.x() - half, center.y() - half)
        gx = round(scene_pos.x() / 12) * 12
        gy = round(scene_pos.y() / 12) * 12
        item.setPos(gx, gy)
        return item

    #def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
    #    if event.button() == QtCore.Qt.MouseButton.LeftButton:
    #        self.browse_for_image()
    #    super().mousePressEvent(event)

    # Delegate DnD to the view to ensure consistent behavior
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        event.ignore()

    _IMG_FILTER = "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*.*)"

    def browse_for_image(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", str(Path.cwd()), self._IMG_FILTER)
        if file_path:
            path = Path(file_path)
            # Place at center when opened from dialog
            self.load_image_from_path(path, None)
            self.imageLoaded.emit(path)

    def browse_for_images(self) -> None:
        """Open several images as a single batch source (one chain, many images)."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Images", str(Path.cwd()), self._IMG_FILTER)
        images = [cv2.imread(p) for p in paths]
        images = [im for im in images if im is not None]
        if not images:
            return
        if len(images) == 1:
            self.add_icon(images[0])
        else:
            self.add_images(images)

    def load_image_from_path(self, path: Path, scene_pos: Optional[QtCore.QPointF] = None) -> None:
        image = cv2.imread(str(path))
        if image is None:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not load image: {path}")
            return
        self.add_icon(image, scene_pos)

    @QtCore.pyqtSlot(Path, QtCore.QPointF)
    def on_file_dropped(self, path: Path, scene_pos: QtCore.QPointF) -> None:
        self.load_image_from_path(path, scene_pos)
        self.imageLoaded.emit(path)

    def add_icon(self, image_bgr, scene_pos: Optional[QtCore.QPointF] = None):
        # Create an ImageNode with the loaded image
        item = ImageNode(image_bgr, icon_size=self.icon_size, grid_size=12)
        self.view.controller.register_source(item, image_bgr)
        self.view._scene.addItem(item)

        # Default drop position to center
        if scene_pos is None:
            rect = self.view._scene.sceneRect()
            half = self.icon_size / 2
            scene_pos = QtCore.QPointF(rect.center().x() - half, rect.center().y() - half)

        # Snap initial position to grid
        gx = round(scene_pos.x() / 12) * 12
        gy = round(scene_pos.y() / 12) * 12
        item.setPos(gx, gy)
        self.instruction.setText("")
        return item

    def add_images(self, images, scene_pos: Optional[QtCore.QPointF] = None):
        """Create one batch source node holding several images."""
        batch = Batch(images)
        item = ImageNode(batch, icon_size=self.icon_size, grid_size=12)
        self.view.controller.register_source(item, batch)
        self.view._scene.addItem(item)
        if scene_pos is None:
            rect = self.view._scene.sceneRect()
            half = self.icon_size / 2
            scene_pos = QtCore.QPointF(rect.center().x() - half, rect.center().y() - half)
        item.setPos(round(scene_pos.x() / 12) * 12, round(scene_pos.y() / 12) * 12)
        self.instruction.setText("")
        return item

    # --- save / load -------------------------------------------------------
    def to_dict(self) -> dict:
        controller = self.view.controller
        positions = {
            gid: (qt.x(), qt.y()) for gid, qt in controller._qt_by_gid.items()
        }
        return persistence.to_dict(controller.model, positions)

    def load_dict(self, data: dict) -> None:
        """Replace the canvas contents with a serialized pipeline."""
        model, positions = persistence.from_dict(data)
        controller = self.view.controller
        self.view._scene.clear()
        controller.adopt(model)

        # Older pipelines were saved with the previous shifting-origin scene, so
        # some node positions are negative. Translate the whole layout into the
        # positive quadrant (relative positions preserved; all-positive layouts are
        # left untouched) so nothing clamps to the pinned (0,0) origin on load.
        if positions:
            margin = 24
            min_x = min(x for x, _ in positions.values())
            min_y = min(y for _, y in positions.values())
            dx = (margin - min_x) if min_x < 0 else 0
            dy = (margin - min_y) if min_y < 0 else 0
            if dx or dy:
                positions = {gid: (x + dx, y + dy) for gid, (x, y) in positions.items()}
            # Give the scene room for the WHOLE layout BEFORE placing any node, so
            # the position clamp in Node.itemChange (which keeps a node inside the
            # scene rect) can't squash a wide pipeline into the default rect.
            max_x = max(x for x, _ in positions.values())
            max_y = max(y for _, y in positions.values())
            self.view._scene.setSceneRect(0, 0, max_x + 600, max_y + 600)

        # Recreate a view item per backend node and bind it.
        qt_by_gid = {}
        for gid, gn in model.nodes.items():
            if gn.is_source:
                item = ImageNode(gn.source_image, icon_size=self.icon_size, grid_size=12)
            else:
                if gn.op.id == "save_to_file":
                    item = SaveToFileNode(icon_size=self.icon_size, grid_size=12)
                elif gn.op.id == "export_code":
                    item = ExportCodeNode(icon_size=self.icon_size, grid_size=12)
                else:
                    item = FunctionNode(gn.op, icon_size=self.icon_size, grid_size=12)
            controller.bind(item, gn)
            self.view._scene.addItem(item)
            x, y = positions.get(gid, (0.0, 0.0))
            item.setPos(x, y)
            qt_by_gid[gid] = item

        # Recreate the arrows for each edge.
        for edge in model.edges:
            src = qt_by_gid.get(edge.src.id)
            dst = qt_by_gid.get(edge.dst.id)
            if src is not None and dst is not None:
                self.view._scene.addItem(ArrowItem(src, dst))

        if model.nodes:
            self.instruction.setText("")
        self.view._update_scene_rect()   # shrink the temporary rect back to fit the content
        controller.recompute_all()

    def resize_all_icons(self, new_size: int) -> None:
        for item in self.view._scene.items():
            if isinstance(item, Node):
                item.set_icon_size(new_size)
                # No need to trigger re-execution for icon size changes
                # The set_icon_size method will handle thumbnail updates


