"""The node canvas: the graphics view and its drop-target wrapper (frontend)."""
from pathlib import Path
from typing import Optional

import cv2
from PyQt6 import QtCore, QtGui, QtWidgets

from core.operations import by_label
from core import persistence
from ui.controller import GraphController
from ui.nodes import Node, ImageNode, FunctionNode, SaveToFileNode
from ui.arrow import ArrowItem

# Default node icon size (px). Owned here since the canvas manages icons.
DEFAULT_ICON_SIZE = 180

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
        # Scene will follow the viewport size to fill the right side
        self._scene.setSceneRect(QtCore.QRectF(self.viewport().rect()))
        # Not scrollable; fill the space provided by the splitter
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Disable built-in zoom/pan; we only place draggable icons
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
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

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        # Keep scene rect matched to available view size
        self._scene.setSceneRect(QtCore.QRectF(0, 0, self.viewport().width(), self.viewport().height()))
        super().resizeEvent(event)

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
        
        # Check if target can accept input from source
        if not target.can_accept_input(source):
            return False
        
        return True
    
    def _get_connection_type(self, source: Optional[Node], target: Optional[Node]) -> str:
        """Get the type of connection: 'valid', 'invalid', or 'implicit_conversion'."""
        if source is None or target is None:
            return 'invalid'
        
        # Check if source is SaveToFile (not allowed as source)
        if isinstance(source, SaveToFileNode):
            return 'invalid'
        
        # Check if target can accept input from source
        if not target.can_accept_input(source):
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
        if source_image is None:
            return False
        
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
        
        # Check if target can accept input from source
        if not target_node.can_accept_input(source_node):
            return  # Target cannot accept this input type or already has max inputs
        
        # Create the arrow and register the connection
        arrow = ArrowItem(source_node, target_node)
        self._scene.addItem(arrow)
        
        # Register the input connection
        target_node.add_input_connection(source_node)

    






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

    def add_function_node(self, label: str, scene_pos: Optional[QtCore.QPointF] = None, meta: Optional[dict] = None):
        # Look the operation up in the registry and build the right node.
        op = by_label.get(label)
        if op is None:
            return None
        if op.id == "save_to_file":
            item = SaveToFileNode(icon_size=self.icon_size, grid_size=12)
        else:
            item = FunctionNode(op, icon_size=self.icon_size, grid_size=12)

        self.view.controller.register_op(item)
        self.view._scene.addItem(item)
        if scene_pos is None:
            rect = self.view._scene.sceneRect()
            half = self.icon_size / 2
            scene_pos = QtCore.QPointF(rect.center().x() - half, rect.center().y() - half)
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

    def browse_for_image(self) -> None:
        filter_str = "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*.*)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", str(Path.cwd()), filter_str)
        if file_path:
            path = Path(file_path)
            # Place at center when opened from dialog
            self.load_image_from_path(path, None)
            self.imageLoaded.emit(path)

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

        # Recreate a view item per backend node and bind it.
        qt_by_gid = {}
        for gid, gn in model.nodes.items():
            if gn.is_source:
                item = ImageNode(gn.source_image, icon_size=self.icon_size, grid_size=12)
            else:
                item = (SaveToFileNode(icon_size=self.icon_size, grid_size=12)
                        if gn.op.id == "save_to_file"
                        else FunctionNode(gn.op, icon_size=self.icon_size, grid_size=12))
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
        controller.recompute_all()

    def resize_all_icons(self, new_size: int) -> None:
        for item in self.view._scene.items():
            if isinstance(item, Node):
                item.set_icon_size(new_size)
                # No need to trigger re-execution for icon size changes
                # The set_icon_size method will handle thumbnail updates


