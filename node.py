import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from PyQt6 import QtCore, QtGui, QtWidgets

from operations import REGISTRY

if TYPE_CHECKING:
    from main import ArrowItem


def cv_to_qimage(image_bgr) -> QtGui.QImage:
    """Convert OpenCV BGR image to QImage."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channel = image_rgb.shape
    bytes_per_line = channel * width
    qimage = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    return qimage.copy()


class Node(QtWidgets.QGraphicsPixmapItem):
    """Base class for all nodes in the visual programming interface."""
    
    def __init__(self, icon_size: int, grid_size: int = 12, meta: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.grid_size = grid_size
        self._icon_size = icon_size
        self._meta = meta or {}
        self._arrows: set['ArrowItem'] = set()
        self._highlighted = False
        self._result_image: Optional[np.ndarray] = None
        self._executing = False
        
        # Setup basic properties
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(1)
        
        # Initialize with placeholder
        self._render_placeholder()
    
    def _render_placeholder(self) -> None:
        """Render a placeholder icon for the node."""
        size = max(1, int(self._icon_size))
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtGui.QColor(200, 200, 200))  # Light gray placeholder
        self.setPixmap(pix)
    
    def set_icon_size(self, icon_size: int) -> None:
        """Update the icon size and re-render."""
        self._icon_size = int(icon_size)
        self._render_icon()
        self._notify_arrows()
    
    def _render_icon(self) -> None:
        """Render the node's icon. Override in subclasses for specific behavior."""
        self._render_placeholder()
    
    def set_highlighted(self, highlighted: bool) -> None:
        """Set the highlighted state of the node."""
        if self._highlighted != highlighted:
            self._highlighted = highlighted
            self.update()
    
    def set_destination_highlighted(self, highlighted: bool, is_valid: bool = True) -> None:
        """Set the destination highlighted state of the node."""
        if not hasattr(self, '_destination_highlighted'):
            self._destination_highlighted = False
        if not hasattr(self, '_destination_valid'):
            self._destination_valid = True
        if not hasattr(self, '_destination_connection_type'):
            self._destination_connection_type = 'valid'
            
        # Handle string connection types (like 'implicit_conversion')
        if isinstance(is_valid, str):
            self._destination_connection_type = is_valid
            is_valid = True  # Still considered valid for highlighting purposes
        else:
            self._destination_connection_type = 'valid' if is_valid else 'invalid'
            
        if self._destination_highlighted != highlighted or self._destination_valid != is_valid:
            self._destination_highlighted = highlighted
            self._destination_valid = is_valid
            self.update()
    
    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionGraphicsItem, widget: Optional[QtWidgets.QWidget] = None) -> None:
        """Paint the node with optional highlighting."""
        super().paint(painter, option, widget)
        
        # Draw source highlight (orange)
        if self._highlighted:
            painter.save()
            try:
                pen = QtGui.QPen(QtGui.QColor(255, 165, 0))  # orange highlight
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawRect(self.boundingRect())
            finally:
                painter.restore()
        
        # Draw destination highlight (green/orange/red)
        if getattr(self, '_destination_highlighted', False):
            painter.save()
            try:
                # Determine color based on connection type
                connection_type = getattr(self, '_destination_connection_type', 'valid')
                if connection_type == 'implicit_conversion':
                    color = QtGui.QColor(255, 165, 0)  # Orange for implicit conversion
                elif getattr(self, '_destination_valid', True):
                    color = QtGui.QColor(0, 255, 0)  # Green for valid
                else:
                    color = QtGui.QColor(255, 0, 0)  # Red for invalid
                
                pen = QtGui.QPen(color)
                pen.setWidth(3)  # Slightly thicker for destination
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawRect(self.boundingRect())
            finally:
                painter.restore()
        
        # Draw type indicator icon
        self._draw_type_icon(painter)
    
    def _draw_type_icon(self, painter: QtGui.QPainter) -> None:
        """Draw a type indicator icon in the top-left corner."""
        # Get the node's bounding rectangle
        rect = self.boundingRect()
        
        # Icon size and position
        icon_size = 16
        margin = 2
        icon_rect = QtCore.QRectF(
            rect.left() + margin,
            rect.top() + margin,
            icon_size,
            icon_size
        )
        
        # Draw background circle
        painter.save()
        try:
            # Semi-transparent background
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 200)))
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 100), 1))
            painter.drawEllipse(icon_rect)
            
            # Draw the specific icon based on node type
            self._draw_specific_type_icon(painter, icon_rect)
        finally:
            painter.restore()
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw the specific icon for this node type. Override in subclasses."""
        # Default: generic node icon (small circle)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
        painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100)))
        center = icon_rect.center()
        painter.drawEllipse(center, 3, 3)
    
    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):
        """Handle position changes with grid snapping."""
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            new_pos: QtCore.QPointF = value
            x = round(new_pos.x() / self.grid_size) * self.grid_size
            y = round(new_pos.y() / self.grid_size) * self.grid_size
            scene = self.scene()
            if scene is not None:
                rect = scene.sceneRect()
                x = max(rect.left(), min(rect.right() - self.pixmap().width(), x))
                y = max(rect.top(), min(rect.bottom() - self.pixmap().height(), y))
            return QtCore.QPointF(x, y)
        elif change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._notify_arrows()
        return super().itemChange(change, value)
    
    def _notify_arrows(self) -> None:
        """Notify all connected arrows to update their paths."""
        for arrow in list(self._arrows):
            try:
                arrow.update_path()
            except RuntimeError:
                # Arrow or scene has been deleted, remove from our list
                self._arrows.discard(arrow)
    
    def _register_arrow(self, arrow: 'ArrowItem') -> None:
        """Register an arrow connection."""
        self._arrows.add(arrow)
    
    def _unregister_arrow(self, arrow: 'ArrowItem') -> None:
        """Unregister an arrow connection."""
        if arrow in self._arrows:
            self._arrows.remove(arrow)
    
    def get_output_image(self) -> Optional[np.ndarray]:
        """Get the output image from this node. Override in subclasses."""
        return self._result_image
    
    def can_accept_input(self, input_node: 'Node') -> bool:
        """Check if this node can accept input from another node. Override in subclasses."""
        return False
    
    def add_input_connection(self, input_node: 'Node') -> bool:
        """Add an input connection. Override in subclasses."""
        return False
    
    def remove_input_connection(self, input_node: 'Node') -> None:
        """Remove an input connection. Override in subclasses."""
        pass
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value. Override in subclasses that have parameters."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters. Override in subclasses."""
        return {}


class ImageNode(Node):
    """Node representing an input image."""
    
    def __init__(self, source_image: np.ndarray, icon_size: int, grid_size: int = 12):
        # Build meta for info panel
        h, w = source_image.shape[:2]
        channels = 1 if source_image.ndim == 2 else source_image.shape[2]
        dtype = source_image.dtype
        if channels == 1:
            kind = "Float" if np.issubdtype(dtype, np.floating) else "Gray"
        else:
            kind = "Float" if np.issubdtype(dtype, np.floating) else "BGR"
        
        meta = {"type": kind, "w": w, "h": h, "channels": channels}
        super().__init__(icon_size, grid_size, meta)
        
        self._source_image = source_image
        self._render_icon()
    
    def _render_icon(self) -> None:
        """Render the image as a thumbnail."""
        size = max(1, int(self._icon_size))
        thumb = QtGui.QPixmap(size, size)
        thumb.fill(QtGui.QColor(255, 255, 255))  # white background
        painter = QtGui.QPainter(thumb)
        try:
            # Convert numpy array to QImage, then to QPixmap
            qimage = cv_to_qimage(self._source_image)
            pix = QtGui.QPixmap.fromImage(qimage)
            scaled = pix.scaled(
                size,
                size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            x = (size - scaled.width()) // 2
            y = (size - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        finally:
            painter.end()
        self.setPixmap(thumb)
    
    def get_output_image(self) -> Optional[np.ndarray]:
        """Return the source image."""
        return self._source_image
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw image icon (photo frame)."""
        painter.save()
        try:
            # Blue color for image nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(33, 150, 243), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            
            # Draw photo frame (rectangle with diagonal lines)
            frame_rect = icon_rect.adjusted(2, 2, -2, -2)
            painter.drawRect(frame_rect)
            
            # Draw diagonal lines (like a photo)
            center = frame_rect.center()
            painter.drawLine(frame_rect.topLeft(), center)
            painter.drawLine(frame_rect.topRight(), center)
            painter.drawLine(frame_rect.bottomLeft(), center)
            painter.drawLine(frame_rect.bottomRight(), center)
        finally:
            painter.restore()


class FunctionNode(Node):
    """A node that runs a registered :class:`operations.Operation`.

    One generic class drives every (pure) operation; the operation supplies
    the ports, parameter defaults, and compute function. This replaces the
    former one-subclass-per-function design.
    """

    def __init__(self, op, icon_size: int, grid_size: int = 12):
        self.op = op
        self._label = op.label
        self._input_connections: List[Node] = []
        self._parameters: Dict[str, Any] = op.defaults()
        self._propagating = False
        meta = {"name": op.id, "in": op.in_label, "out": op.out_label}
        super().__init__(icon_size, grid_size, meta)

    def set_icon_size(self, icon_size: int) -> None:
        """Update the icon size and re-render."""
        self._icon_size = int(icon_size)
        # Flag prevents propagation while we are only resizing thumbnails.
        self._resizing_icons = True
        if self._result_image is not None:
            self._update_result_thumbnail()
        else:
            self._render_icon()
        self._notify_arrows()
        self._resizing_icons = False

    def _render_icon(self) -> None:
        """Render the function node icon (label + funnel glyph)."""
        size = max(1, int(self._icon_size))
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtGui.QColor(255, 255, 255))
        painter = QtGui.QPainter(pix)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 255, 200)))
            painter.drawRect(1, 1, size - 2, size - 2)

            # Small funnel glyph in the top-left.
            glyph_margin = max(3, int(size * 0.06))
            top = glyph_margin
            left = glyph_margin
            funnel_width = max(8, int(size * 0.22))
            funnel_height = max(6, int(size * 0.16))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 120, 0)))
            painter.drawRect(left, top, funnel_width, int(funnel_height * 0.45))
            tri = QtGui.QPolygonF([
                QtCore.QPointF(left, top + int(funnel_height * 0.45)),
                QtCore.QPointF(left + funnel_width, top + int(funnel_height * 0.45)),
                QtCore.QPointF(left + funnel_width / 2.0, top + funnel_height),
            ])
            painter.drawPolygon(tri)

            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
            font = painter.font()
            font.setPointSize(max(7, int(size * 0.18)))
            font.setBold(True)
            painter.setFont(font)
            rect = QtCore.QRectF(4, 4, size - 8, size - 8)
            flags = (
                QtCore.Qt.AlignmentFlag.AlignCenter
                | QtCore.Qt.TextFlag.TextWordWrap
            )
            painter.drawText(rect, flags, self._label)
        finally:
            painter.end()
        self.setPixmap(pix)

    def can_accept_input(self, input_node: Node) -> bool:
        """Accept inputs until every input port of the operation is filled."""
        return len(self._input_connections) < len(self.op.inputs)

    def add_input_connection(self, input_node: Node) -> bool:
        """Add an input connection if a port is still free."""
        if self.can_accept_input(input_node):
            self._input_connections.append(input_node)
            self._check_and_execute()
            return True
        return False

    def remove_input_connection(self, input_node: Node) -> None:
        """Remove an input connection."""
        if input_node in self._input_connections:
            self._input_connections.remove(input_node)

    def _check_and_execute(self) -> None:
        """Run the operation once all input ports are connected and have data."""
        if self._executing or getattr(self, '_propagating', False):
            return
        if len(self._input_connections) != len(self.op.inputs):
            return
        inputs = [c.get_output_image() for c in self._input_connections]
        if any(img is None for img in inputs):
            return

        self._executing = True
        try:
            result_image = self._compute(inputs)
            if result_image is not None:
                self._result_image = result_image
                self._update_result_thumbnail()
        finally:
            self._executing = False

    def _compute(self, inputs: List[np.ndarray]) -> Optional[np.ndarray]:
        """Invoke the operation's compute function. Subclasses may override."""
        return self.op.compute(inputs, self._parameters)

    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Update a parameter and re-execute (with downstream propagation)."""
        if param_name not in self._parameters:
            return
        self._parameters[param_name] = value
        if len(self._input_connections) != len(self.op.inputs):
            return
        if preview_mode:
            self._re_execute_preview()
        else:
            self._check_and_execute()
            if not getattr(self, '_propagating', False):
                self._propagate_changes()

    def get_parameters(self) -> Dict[str, Any]:
        """Return a copy of the current parameter values."""
        return self._parameters.copy()

    def _update_result_thumbnail(self) -> None:
        """Update the node's thumbnail to show the computed result."""
        if self._result_image is None:
            return

        qimage = cv_to_qimage(self._result_image)
        pix = QtGui.QPixmap.fromImage(qimage)

        size = max(1, int(self._icon_size))
        thumb = QtGui.QPixmap(size, size)
        thumb.fill(QtGui.QColor(255, 255, 255))
        painter = QtGui.QPainter(thumb)
        try:
            scaled = pix.scaled(
                size, size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            x = (size - scaled.width()) // 2
            y = (size - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            pen = QtGui.QPen(QtGui.QColor(0, 0, 255))  # blue border = function result
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(1, 1, size - 2, size - 2)
        finally:
            painter.end()

        self.setPixmap(thumb)

        # Propagate only on a real (committed) recompute.
        if (not getattr(self, '_preview_mode', False)
                and not self._executing
                and not getattr(self, '_resizing_icons', False)):
            self._propagate_changes()

    def get_output_image(self) -> Optional[np.ndarray]:
        """Return the result image."""
        return self._result_image

    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw a generic marker tinted with the operation's color."""
        painter.save()
        try:
            r, g, b = self.op.color
            painter.setPen(QtGui.QPen(QtGui.QColor(r, g, b), 1))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
            painter.drawEllipse(icon_rect.center(), 4, 4)
        finally:
            painter.restore()

    def _re_execute_preview(self) -> None:
        """Re-execute in preview mode (update thumbnail only; no file writes / propagation)."""
        self._preview_mode = True
        self._check_and_execute()
        self._preview_mode = False

    def _propagate_changes(self) -> None:
        """Re-execute downstream function nodes connected via arrows."""
        if getattr(self, '_propagating', False):
            return
        self._propagating = True
        try:
            scene = self.scene()
            if scene is None:
                return
            for item in scene.items():
                if item.__class__.__name__ != 'ArrowItem':
                    continue
                if not (hasattr(item, 'a') and hasattr(item, 'b')):
                    continue
                if item.a is not self and item.b is not self:
                    continue
                target = item.b if item.a is self else item.a
                if isinstance(target, FunctionNode):
                    if getattr(self, '_preview_mode', False):
                        target._preview_mode = True
                    target._check_and_execute()
                    if not isinstance(target, SaveToFileNode):
                        target._propagate_changes()
        except RuntimeError:
            # Scene or items have been deleted; ignore.
            pass
        finally:
            self._propagating = False


class SaveToFileNode(FunctionNode):
    """Saves the input image to disk. Side-effecting and stateful, so it keeps
    its own compute rather than using a pure registry function."""

    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__(REGISTRY["save_to_file"], icon_size, grid_size)

    def _compute(self, inputs: List[np.ndarray]) -> Optional[np.ndarray]:
        """Write the input image to ./output and return it unchanged for chaining."""
        import os
        import time

        input_image = inputs[0]

        # Never write while propagating or during a parameter preview (slider drag).
        if getattr(self, '_propagating', False) or getattr(self, '_preview_mode', False):
            return input_image

        try:
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)

            if not hasattr(self, '_process_start_time'):
                self._process_start_time = time.time()
            if not hasattr(self, '_node_index'):
                self._node_index = id(self) % 10000

            process_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._process_start_time))
            default_filename = f"save_to_file_{process_timestamp}_{self._node_index}.png"

            filename = self._parameters.get("filename", "")
            use_custom = self._parameters.get("use_custom", False)
            if not use_custom or not filename:
                filename = default_filename

            if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
                filename += '.png'

            filepath = os.path.join(output_dir, filename)
            if cv2.imwrite(filepath, input_image):
                print(f"Image saved to: {filepath}")
                return input_image
            print(f"Failed to save image to: {filepath}")
            return None
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
