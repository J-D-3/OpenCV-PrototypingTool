import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from node import Node, ImageNode, FunctionNode, SaveToFileNode, BlurNode, ThresholdNode, ToGrayscaleNode, ToBGRNode, AdaptiveThresholdNode, SumNode, AndNode, DiffNode, MSERNode, cv_to_qimage

# Default icon size constant
DEFAULT_ICON_SIZE = 180


class ImageViewerWindow(QtWidgets.QMainWindow):
    """Window for displaying node result images with scaling and scrolling."""
    
    def __init__(self, node: Node, parent=None):
        super().__init__(parent)
        self.node = node
        self.setWindowTitle(f"Image Viewer - {node._meta.get('name', 'Node')}")
        self.setMinimumSize(400, 300)
        
        # Create central widget with scroll area
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Create scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create label for image display
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)  # We'll handle scaling manually
        self.image_label.setMinimumSize(100, 100)
        
        # Set the label as the scroll area's widget
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)
        
        # Add zoom controls
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
        
        # Initialize zoom level
        self.zoom_factor = 1.0
        self.original_image = None
        
        # Connect to node changes
        self.connect_to_node()
        
        # Update the image
        self.update_image()
    
    def connect_to_node(self):
        """Connect to the node's changes to update the image automatically."""
        # We'll use a timer to check for changes periodically
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.check_for_updates)
        self.update_timer.start(100)  # Check every 100ms
    
    def check_for_updates(self):
        """Check if the node's output has changed and update if necessary."""
        if self.node is None:
            return
        
        # Get the current output image
        current_image = self.node.get_output_image()
        
        # Check if the image has changed
        if current_image is not None and not self.images_equal(current_image, self.original_image):
            self.original_image = current_image.copy() if current_image is not None else None
            self.update_image()
    
    def images_equal(self, img1, img2):
        """Check if two images are equal."""
        if img1 is None and img2 is None:
            return True
        if img1 is None or img2 is None:
            return False
        if img1.shape != img2.shape:
            return False
        return (img1 == img2).all()
    
    def update_image(self):
        """Update the displayed image."""
        if self.node is None:
            return
        
        # Get the output image from the node
        output_image = self.node.get_output_image()
        
        if output_image is None:
            self.image_label.setText("No image available")
            self.image_label.setPixmap(QtGui.QPixmap())
            return
        
        # Convert to QImage
        qimage = cv_to_qimage(output_image)
        
        # Convert to QPixmap
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        # Store original for zooming
        self.original_pixmap = pixmap
        
        # Apply current zoom
        self.apply_zoom()
    
    def apply_zoom(self):
        """Apply the current zoom factor to the image."""
        if not hasattr(self, 'original_pixmap') or self.original_pixmap.isNull():
            return
        
        # Calculate new size
        original_size = self.original_pixmap.size()
        new_size = QtCore.QSize(
            int(original_size.width() * self.zoom_factor),
            int(original_size.height() * self.zoom_factor)
        )
        
        # Scale the pixmap
        scaled_pixmap = self.original_pixmap.scaled(
            new_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        
        # Update the label
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        
        # Update zoom label
        self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")
    
    def zoom_in(self):
        """Increase zoom level."""
        self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)  # Max 500%
        self.apply_zoom()
    
    def zoom_out(self):
        """Decrease zoom level."""
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)  # Min 10%
        self.apply_zoom()
    
    def fit_to_window(self):
        """Fit the image to the window size."""
        if not hasattr(self, 'original_pixmap') or self.original_pixmap.isNull():
            return
        
        # Get the available space in the scroll area
        scroll_area = self.image_label.parent().parent()  # Get the scroll area
        available_size = scroll_area.size()
        
        # Calculate zoom factor to fit
        original_size = self.original_pixmap.size()
        scale_x = available_size.width() / original_size.width()
        scale_y = available_size.height() / original_size.height()
        self.zoom_factor = min(scale_x, scale_y) * 0.9  # 90% to leave some margin
        
        self.apply_zoom()
    
    def closeEvent(self, event):
        """Clean up when the window is closed."""
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        super().closeEvent(event)


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
            if isinstance(target, (ThresholdNode, AdaptiveThresholdNode)):
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

    





class ArrowItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, a: Node, b: Node):
        super().__init__()
        self.a = a
        self.b = b
        # Register with endpoints so they can notify us when moving/resizing
        self.a._register_arrow(self)
        self.b._register_arrow(self)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
        pen.setWidth(2)
        self.setPen(pen)
        # Ensure arrows render below icons
        self.setZValue(0)
        self.update_path()

    def update_path(self) -> None:
        try:
            a_rect = self.a.sceneBoundingRect()
            b_rect = self.b.sceneBoundingRect()
            a_center = a_rect.center()
            b_center = b_rect.center()
            p = a_center
            # Compute intersection of line (from b_center toward a_center) with b_rect edge
            dx = a_center.x() - b_center.x()
            dy = a_center.y() - b_center.y()
            # Avoid degenerate vector
            if dx == 0 and dy == 0:
                q = b_center
            else:
                # t such that b_center + t*(dx,dy) hits the rectangle boundary (t > 0)
                tx = float('inf')
                ty = float('inf')
                if dx != 0:
                    tx = ((b_rect.right() - b_center.x()) / dx) if dx > 0 else ((b_rect.left() - b_center.x()) / dx)
                if dy != 0:
                    ty = ((b_rect.bottom() - b_center.y()) / dy) if dy > 0 else ((b_rect.top() - b_center.y()) / dy)
                t = min(t for t in (tx, ty) if t > 0)
                q = QtCore.QPointF(b_center.x() + dx * t, b_center.y() + dy * t)
            path = QtGui.QPainterPath(p)
            path.lineTo(q)
            # Append arrow head at q
            head = self._arrow_head(p, q, size=8.0)
            path.addPath(head)
            self.setPath(path)
        except RuntimeError:
            # Scene or items have been deleted, ignore
            pass

    def _arrow_head(self, p: QtCore.QPointF, q: QtCore.QPointF, size: float) -> QtGui.QPainterPath:
        import math
        angle = math.atan2(q.y() - p.y(), q.x() - p.x())
        left = QtCore.QPointF(
            q.x() - size * math.cos(angle - math.pi / 6.0),
            q.y() - size * math.sin(angle - math.pi / 6.0),
        )
        right = QtCore.QPointF(
            q.x() - size * math.cos(angle + math.pi / 6.0),
            q.y() - size * math.sin(angle + math.pi / 6.0),
        )
        head = QtGui.QPainterPath(q)
        head.lineTo(left)
        head.lineTo(right)
        head.closeSubpath()
        return head


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

    def add_function_node(self, label: str, scene_pos: Optional[QtCore.QPointF] = None, meta: Optional[dict] = None) -> None:
        # Create the appropriate function node based on the label
        if label == "Save to File":
            item = SaveToFileNode(icon_size=self.icon_size, grid_size=12)
        elif label == "Blur":
            item = BlurNode(icon_size=self.icon_size, grid_size=12)
        elif label == "Threshold":
            item = ThresholdNode(icon_size=self.icon_size, grid_size=12)
        elif label == "Adaptive Threshold":
            item = AdaptiveThresholdNode(icon_size=self.icon_size, grid_size=12)
        elif label == "To Grayscale":
            item = ToGrayscaleNode(icon_size=self.icon_size, grid_size=12)
        elif label == "To BGR":
            item = ToBGRNode(icon_size=self.icon_size, grid_size=12)
        elif label == "Sum":
            item = SumNode(icon_size=self.icon_size, grid_size=12)
        elif label == "AND":
            item = AndNode(icon_size=self.icon_size, grid_size=12)
        elif label == "Diff":
            item = DiffNode(icon_size=self.icon_size, grid_size=12)
        elif label == "MSER":
            item = MSERNode(icon_size=self.icon_size, grid_size=12)
        else:
            # Fallback to generic function node
            item = FunctionNode(label, icon_size=self.icon_size, grid_size=12, meta=meta)
        
        self.view._scene.addItem(item)
        if scene_pos is None:
            rect = self.view._scene.sceneRect()
            half = self.icon_size / 2
            scene_pos = QtCore.QPointF(rect.center().x() - half, rect.center().y() - half)
        gx = round(scene_pos.x() / 12) * 12
        gy = round(scene_pos.y() / 12) * 12
        item.setPos(gx, gy)

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

    def add_icon(self, image_bgr, scene_pos: Optional[QtCore.QPointF] = None) -> None:
        # Create an ImageNode with the loaded image
        item = ImageNode(image_bgr, icon_size=self.icon_size, grid_size=12)
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

    def resize_all_icons(self, new_size: int) -> None:
        for item in self.view._scene.items():
            if isinstance(item, Node):
                item.set_icon_size(new_size)
                # No need to trigger re-execution for icon size changes
                # The set_icon_size method will handle thumbnail updates


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
        open_btn = QtWidgets.QPushButton("Open Image…")
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

        categories = [
            "Input/Output",
            "Conversions",
            "Geometry",
            "Arithmetic Operations",
            "Local Operations",
            "Fourier",
        ]
        category_items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for cat in categories:
            item = QtWidgets.QTreeWidgetItem([cat])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            tree.addTopLevelItem(item)
            category_items[cat] = item

        # Functions with simple metadata
        # Input/Output
        save_to_file_item = QtWidgets.QTreeWidgetItem(["Save to File"])
        save_to_file_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "save_to_file", "in": "Mat (Any)", "out": "File"})
        category_items["Input/Output"].addChild(save_to_file_item)
        
        # Local Operations
        blur_item = QtWidgets.QTreeWidgetItem(["Blur"])
        blur_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "blur", "in": "Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        category_items["Local Operations"].addChild(blur_item)

        thresh_item = QtWidgets.QTreeWidgetItem(["Threshold"])
        thresh_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "threshold", "in": "Mat (Gray)", "out": "Mat (Binary/Gray)"})
        category_items["Local Operations"].addChild(thresh_item)
        
        adaptive_thresh_item = QtWidgets.QTreeWidgetItem(["Adaptive Threshold"])
        adaptive_thresh_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "adaptive_threshold", "in": "Mat (Gray)", "out": "Mat (Binary)"})
        category_items["Local Operations"].addChild(adaptive_thresh_item)
        
        mser_item = QtWidgets.QTreeWidgetItem(["MSER"])
        mser_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "mser", "in": "Mat (Gray)", "out": "Mat (BGR)"})
        category_items["Local Operations"].addChild(mser_item)
        
        # Conversions
        to_gray_item = QtWidgets.QTreeWidgetItem(["To Grayscale"])
        to_gray_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "to_grayscale", "in": "Mat (BGR)", "out": "Mat (Gray)"})
        category_items["Conversions"].addChild(to_gray_item)
        
        to_bgr_item = QtWidgets.QTreeWidgetItem(["To BGR"])
        to_bgr_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "to_bgr", "in": "Mat (Gray)", "out": "Mat (BGR)"})
        category_items["Conversions"].addChild(to_bgr_item)
        
        # Arithmetic Operations
        sum_item = QtWidgets.QTreeWidgetItem(["Sum"])
        sum_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "sum", "in": "Mat (BGR/Gray) + Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        category_items["Arithmetic Operations"].addChild(sum_item)
        
        and_item = QtWidgets.QTreeWidgetItem(["AND"])
        and_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "and", "in": "Mat (BGR/Gray) & Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        category_items["Arithmetic Operations"].addChild(and_item)
        
        diff_item = QtWidgets.QTreeWidgetItem(["Diff"])
        diff_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"name": "diff", "in": "Mat (BGR/Gray) - Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        category_items["Arithmetic Operations"].addChild(diff_item)

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
                            f"Size: {meta.get('w','?')}×{meta.get('h','?')}\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV image viewer (PyQt6) with sidebar and drag-and-drop pane.")
    parser.add_argument("image_path", nargs="?", default=None, help="Optional path to an initial image")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    initial_image = None
    window_title = "OpenCV Image Viewer"

    if args.image_path is not None:
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"Error: File not found: '{image_path}'.", file=sys.stderr)
            sys.exit(1)
        initial_image = cv2.imread(str(image_path))
        if initial_image is None:
            print(f"Error: Could not load image from '{image_path}'. Check the path and file format.", file=sys.stderr)
            sys.exit(1)
        window_title = f"Image - {image_path.name}"

    window = MainWindow(initial_image, window_title)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()