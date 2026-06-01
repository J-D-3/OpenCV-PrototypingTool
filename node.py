import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from PyQt6 import QtCore, QtGui, QtWidgets

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
    """Base class for function nodes that process images."""
    
    def __init__(self, label: str, icon_size: int, grid_size: int = 12, meta: Optional[Dict[str, Any]] = None):
        self._label = label
        self._input_connections: List[Node] = []
        self._parameters: Dict[str, Any] = {}
        
        # Set up meta with function info
        if meta is None:
            meta = {"name": label, "in": "Mat", "out": "Mat"}
        super().__init__(icon_size, grid_size, meta)
    
    def set_icon_size(self, icon_size: int) -> None:
        """Update the icon size and re-render."""
        self._icon_size = int(icon_size)
        # Set flag to prevent propagation during icon resizing
        self._resizing_icons = True
        # If we have a result image, update the result thumbnail
        if self._result_image is not None:
            self._update_result_thumbnail()
        else:
            # Otherwise, just render the function icon
            self._render_icon()
        self._notify_arrows()
        # Clear the flag
        self._resizing_icons = False
    
    def _render_icon(self) -> None:
        """Render the function node icon."""
        size = max(1, int(self._icon_size))
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtGui.QColor(255, 255, 255))
        painter = QtGui.QPainter(pix)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            # Draw border to differentiate function nodes
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(200, 255, 200)))
            painter.drawRect(1, 1, size - 2, size - 2)

            # Draw a small glyph in the top-left (simple funnel icon)
            glyph_margin = max(3, int(size * 0.06))
            top = glyph_margin
            left = glyph_margin
            funnel_width = max(8, int(size * 0.22))
            funnel_height = max(6, int(size * 0.16))
            # Rectangle (input bar)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 120, 0)))
            painter.drawRect(left, top, funnel_width, int(funnel_height * 0.45))
            # Triangle (funnel)
            tri = QtGui.QPolygonF([
                QtCore.QPointF(left, top + int(funnel_height * 0.45)),
                QtCore.QPointF(left + funnel_width, top + int(funnel_height * 0.45)),
                QtCore.QPointF(left + funnel_width / 2.0, top + funnel_height),
            ])
            painter.drawPolygon(tri)

            # Draw text centered, wrapping if needed
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
        """Check if this function can accept the given input."""
        # Default implementation - can accept one input from any node type
        return len(self._input_connections) == 0
    
    def add_input_connection(self, input_node: Node) -> bool:
        """Add an input connection if possible."""
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
        """Execute the function if all required inputs are available."""
        if self._executing:
            return
        self._executing = True
        
        try:
            if len(self._input_connections) == 0:
                return
            
            # Get the input image from the connected node
            input_node = self._input_connections[0]
            input_image = input_node.get_output_image()
            if input_image is None:
                return
            
            # Execute the function
            result_image = self._execute_function(input_image)
            if result_image is not None:
                self._result_image = result_image
                self._update_result_thumbnail()
        finally:
            self._executing = False
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Execute the specific function. Override in subclasses."""
        return input_image
    
    def _update_result_thumbnail(self) -> None:
        """Update the function node's thumbnail to show the computed result."""
        if self._result_image is None:
            return
        
        # Convert result to QPixmap and create thumbnail
        qimage = cv_to_qimage(self._result_image)
        pix = QtGui.QPixmap.fromImage(qimage)
        
        # Create thumbnail with the same size as icon
        size = max(1, int(self._icon_size))
        thumb = QtGui.QPixmap(size, size)
        thumb.fill(QtGui.QColor(255, 255, 255))
        painter = QtGui.QPainter(thumb)
        try:
            # Draw the result image scaled to fit
            scaled = pix.scaled(size, size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            x = (size - scaled.width()) // 2
            y = (size - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            
            # Draw a border to indicate this is a function result
            pen = QtGui.QPen(QtGui.QColor(0, 0, 255))  # Blue border for function results
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(1, 1, size - 2, size - 2)
        finally:
            painter.end()
        
        self.setPixmap(thumb)
        
        # Only propagate changes if we're not in preview mode, not currently executing, and not resizing icons
        if not getattr(self, '_preview_mode', False) and not self._executing and not getattr(self, '_resizing_icons', False):
            self._propagate_changes()
    
    def get_output_image(self) -> Optional[np.ndarray]:
        """Return the result image."""
        return self._result_image
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw save icon (floppy disk)."""
        painter.save()
        try:
            # Green color for save nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(76, 175, 80), 2))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(76, 175, 80, 50)))
            
            # Draw floppy disk shape
            disk_rect = icon_rect.adjusted(2, 2, -2, -2)
            painter.drawRect(disk_rect)
            
            # Draw disk label (small rectangle in center)
            label_rect = disk_rect.adjusted(3, 3, -3, -3)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(76, 175, 80)))
            painter.drawRect(label_rect)
            
            # Draw arrow pointing down
            center = disk_rect.center()
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
            painter.drawLine(QtCore.QPointF(center.x() - 2, center.y() - 1), QtCore.QPointF(center.x(), center.y() + 1))
            painter.drawLine(QtCore.QPointF(center.x(), center.y() + 1), QtCore.QPointF(center.x() + 2, center.y() - 1))
        finally:
            painter.restore()
    
    def _re_execute_preview(self) -> None:
        """Re-execute function in preview mode (update thumbnails but don't write files)."""
        # Set preview mode flag to prevent file writing
        self._preview_mode = True
        self._check_and_execute()
        # Don't propagate changes during preview mode to prevent infinite loops
        # The preview mode flag will be passed to downstream nodes when they execute
        self._preview_mode = False
    
    def _propagate_changes(self) -> None:
        """Propagate changes to all downstream function nodes."""
        # Prevent infinite loops during propagation
        if getattr(self, '_propagating', False):
            return
        self._propagating = True
        
        try:
            # Find all arrows that start from this function node
            scene = self.scene()
            if scene is None:
                return
                
            for item in scene.items():
                if hasattr(item, '__class__') and item.__class__.__name__ == 'ArrowItem':
                    # Check if this arrow starts from this function node
                    if hasattr(item, 'a') and hasattr(item, 'b'):
                        if item.a == self or item.b == self:
                            # Find the target function node
                            target = item.b if item.a == self else item.a
                            if isinstance(target, FunctionNode):
                                # Pass the preview mode flag to downstream functions
                                if hasattr(self, '_preview_mode') and self._preview_mode:
                                    target._preview_mode = True
                                # Re-execute the target function
                                target._check_and_execute()
                                # Don't propagate further from save nodes to prevent infinite loops
                                if not isinstance(target, SaveToFileNode):
                                    target._propagate_changes()
        except RuntimeError:
            # Scene or items have been deleted, ignore
            pass
        finally:
            self._propagating = False


class SaveToFileNode(FunctionNode):
    """Node for saving images to files."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("Save to File", icon_size, grid_size, {"name": "save_to_file", "in": "Mat (Any)", "out": "File"})
        self._parameters = {
            "filename": "",
            "use_custom": False
        }
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Save the image to a file and return the original image for chaining."""
        import os
        import time
        
        # Check if we're in propagation mode to prevent infinite saves
        if getattr(self, '_propagating', False):
            return input_image  # Just return the image without saving
        
        # Check if we're in preview mode to prevent file writing during slider dragging
        if getattr(self, '_preview_mode', False):
            return input_image  # Just return the image without saving
        
        try:
            # Create output directory if it doesn't exist
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Get or create process start time and node index
            if not hasattr(self, '_process_start_time'):
                self._process_start_time = time.time()
            if not hasattr(self, '_node_index'):
                # Generate a unique index for this save_to_file node
                self._node_index = id(self) % 10000  # Use object ID as unique index
            
            # Generate filename with process start time and node index
            process_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._process_start_time))
            default_filename = f"save_to_file_{process_timestamp}_{self._node_index}.png"
            
            # Get filename from parameters
            filename = self._parameters.get("filename", "")
            use_custom = self._parameters.get("use_custom", False)
            
            if not use_custom or not filename:
                filename = default_filename
            
            # Ensure filename has extension
            if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
                filename += '.png'
            
            # Full path
            filepath = os.path.join(output_dir, filename)
            
            # Save the image
            success = cv2.imwrite(filepath, input_image)
            if success:
                print(f"Image saved to: {filepath}")
                return input_image  # Return original for chaining
            else:
                print(f"Failed to save image to: {filepath}")
                return None
                
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) > 0:
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()


class BlurNode(FunctionNode):
    """Node for blurring images."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("Blur", icon_size, grid_size, {"name": "blur", "in": "Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        self._parameters = {"kernel_size": 15}
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Apply blur to the input image."""
        try:
            kernel_size = self._parameters["kernel_size"]
            return cv2.blur(input_image, (kernel_size, kernel_size))
            
        except Exception as e:
            print(f"Error executing blur: {e}")
            return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) > 0:
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw blur icon (concentric circles)."""
        painter.save()
        try:
            # Purple color for blur nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(156, 39, 176), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            
            # Draw concentric circles (blur effect)
            center = icon_rect.center()
            for i in range(3):
                radius = 3 + i * 2
                painter.drawEllipse(center, radius, radius)
        finally:
            painter.restore()


class ThresholdNode(FunctionNode):
    """Node for thresholding images."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("Threshold", icon_size, grid_size, {"name": "threshold", "in": "Mat (Gray)", "out": "Mat (Binary/Gray)"})
        self._parameters = {
            "threshold_value": 127,
            "max_value": 255,
            "threshold_type": cv2.THRESH_BINARY
        }
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Apply threshold to the input image."""
        try:
            if len(input_image.shape) == 3:
                gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_image
            
            params = self._parameters
            _, result = cv2.threshold(gray, params["threshold_value"], params["max_value"], params["threshold_type"])
            return result
        except Exception as e:
            print(f"Error executing threshold: {e}")
            return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) > 0:
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw threshold icon (binary squares)."""
        painter.save()
        try:
            # Orange color for threshold nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 152, 0), 2))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 152, 0, 100)))
            
            # Draw binary pattern (black and white squares)
            square_size = 3
            for i in range(2):
                for j in range(2):
                    x = icon_rect.left() + 2 + j * (square_size + 1)
                    y = icon_rect.top() + 2 + i * (square_size + 1)
                    square_rect = QtCore.QRectF(x, y, square_size, square_size)
                    if (i + j) % 2 == 0:
                        painter.fillRect(square_rect, QtGui.QColor(0, 0, 0))
                    else:
                        painter.fillRect(square_rect, QtGui.QColor(255, 255, 255))
        finally:
            painter.restore()


class ToGrayscaleNode(FunctionNode):
    """Node for converting images to grayscale."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("To Grayscale", icon_size, grid_size, {"name": "to_grayscale", "in": "Mat (BGR)", "out": "Mat (Gray)"})
        self._parameters = {}
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Convert image to grayscale."""
        try:
            if len(input_image.shape) == 3:
                return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            else:
                return input_image  # Already grayscale
        except Exception as e:
            print(f"Error executing to_grayscale: {e}")
            return None
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw grayscale icon (color wheel with gradient)."""
        painter.save()
        try:
            # Gray color for grayscale nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(96, 96, 96), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            
            # Draw color wheel with grayscale gradient
            center = icon_rect.center()
            radius = 5
            painter.drawEllipse(center, radius, radius)
            
            # Draw gradient lines from center
            for i in range(4):
                angle = i * 90
                x = center.x() + radius * 0.7 * np.cos(np.radians(angle))
                y = center.y() + radius * 0.7 * np.sin(np.radians(angle))
                painter.drawLine(center, QtCore.QPointF(x, y))
        finally:
            painter.restore()


class ToBGRNode(FunctionNode):
    """Node for converting images to BGR."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("To BGR", icon_size, grid_size, {"name": "to_bgr", "in": "Mat (Gray)", "out": "Mat (BGR)"})
        self._parameters = {}
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Convert image to BGR."""
        try:
            if len(input_image.shape) == 2:
                return cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            else:
                return input_image  # Already BGR
        except Exception as e:
            print(f"Error executing to_bgr: {e}")
            return None
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw BGR icon (color wheel with RGB colors)."""
        painter.save()
        try:
            # Blue color for BGR nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(33, 150, 243), 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            
            # Draw color wheel with RGB segments
            center = icon_rect.center()
            radius = 5
            
            # Draw RGB color segments
            colors = [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)]
            for i, color in enumerate(colors):
                start_angle = i * 120
                painter.setBrush(QtGui.QBrush(color))
                painter.drawPie(int(center.x() - radius), int(center.y() - radius), int(radius * 2), int(radius * 2), start_angle * 16, 120 * 16)
        finally:
            painter.restore()


class AdaptiveThresholdNode(FunctionNode):
    """Node for adaptive thresholding images."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("Adaptive Threshold", icon_size, grid_size, {"name": "adaptive_threshold", "in": "Mat (Gray)", "out": "Mat (Binary)"})
        self._parameters = {
            "max_value": 255,
            "adaptive_method": cv2.ADAPTIVE_THRESH_MEAN_C,
            "threshold_type": cv2.THRESH_BINARY,
            "block_size": 11,
            "c": 2
        }
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Apply adaptive threshold to the input image."""
        try:
            if len(input_image.shape) == 3:
                gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_image
            
            params = self._parameters
            result = cv2.adaptiveThreshold(
                gray,
                params["max_value"],
                params["adaptive_method"],
                params["threshold_type"],
                params["block_size"],
                params["c"]
            )
            return result
        except Exception as e:
            print(f"Error executing adaptive_threshold: {e}")
            return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) > 0:
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
import cv2
import numpy as np
from typing import Optional, Dict, Any
from PyQt6 import QtCore, QtGui, QtWidgets
from node import FunctionNode, Node


class SumNode(FunctionNode):
    """Node for summing two images using cv2.addWeighted."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("Sum", icon_size, grid_size, {"name": "sum", "in": "Mat (BGR/Gray) + Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        self._parameters = {
            "alpha": 0.5
        }
        # Initialize missing attributes from base class
        self._propagating = False
    
    def can_accept_input(self, input_node: Node) -> bool:
        """Check if this function can accept the given input."""
        # Sum node can accept up to 2 inputs
        return len(self._input_connections) < 2
    
    def _check_and_execute(self) -> None:
        """Override to handle two-input execution for SumNode."""
        if self._executing or self._propagating:
            return
        
        # SumNode needs exactly 2 inputs to execute
        if len(self._input_connections) != 2:
            return
        
        # Check if both inputs have valid images
        input1 = self._input_connections[0].get_output_image()
        input2 = self._input_connections[1].get_output_image()
        
        if input1 is None or input2 is None:
            return
        
        self._executing = True
        try:
            # Perform the weighted sum
            result = self._execute_sum(input1, input2)
            if result is not None:
                self._result_image = result
                self._update_result_thumbnail()
        finally:
            self._executing = False
    
    def _execute_sum(self, input1: np.ndarray, input2: np.ndarray) -> Optional[np.ndarray]:
        """Sum the two input images using cv2.addWeighted."""
        try:
            # Ensure both images have the same dimensions and type
            if input1.shape != input2.shape:
                # Resize the second image to match the first
                input2 = cv2.resize(input2, (input1.shape[1], input1.shape[0]))
            
            # Get alpha parameter
            alpha = self._parameters.get("alpha", 0.5)
            beta = 1.0 - alpha
            
            # Perform weighted sum
            result = cv2.addWeighted(input1, alpha, input2, beta, 0)
            return result
            
        except Exception as e:
            print(f"Error executing sum: {e}")
            return None
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """This method is not used for SumNode since it needs two inputs."""
        # SumNode uses _execute_sum instead
        return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) == 2:  # Only execute if we have both inputs
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw sum icon (plus sign)."""
        painter.save()
        try:
            # Purple color for sum nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(128, 0, 128), 2))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(128, 0, 128, 50)))
            
            # Draw plus sign
            center = icon_rect.center()
            size = 4
            
            # Horizontal line
            painter.drawLine(
                QtCore.QPointF(center.x() - size, center.y()),
                QtCore.QPointF(center.x() + size, center.y())
            )
            # Vertical line
            painter.drawLine(
                QtCore.QPointF(center.x(), center.y() - size),
                QtCore.QPointF(center.x(), center.y() + size)
            )
        finally:
            painter.restore()


class AndNode(FunctionNode):
    """Node for bitwise AND operation between two images using cv2.bitwise_and."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("AND", icon_size, grid_size, {"name": "and", "in": "Mat (BGR/Gray) & Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        self._parameters = {}
        # Initialize missing attributes from base class
        self._propagating = False
    
    def can_accept_input(self, input_node: Node) -> bool:
        """Check if this function can accept the given input."""
        # AND node can accept up to 2 inputs
        return len(self._input_connections) < 2
    
    def _check_and_execute(self) -> None:
        """Override to handle two-input execution for AndNode."""
        if self._executing or self._propagating:
            return
        
        # AndNode needs exactly 2 inputs to execute
        if len(self._input_connections) != 2:
            return
        
        # Check if both inputs have valid images
        input1 = self._input_connections[0].get_output_image()
        input2 = self._input_connections[1].get_output_image()
        
        if input1 is None or input2 is None:
            return
        
        self._executing = True
        try:
            # Perform the bitwise AND
            result = self._execute_and(input1, input2)
            if result is not None:
                self._result_image = result
                self._update_result_thumbnail()
        finally:
            self._executing = False
    
    def _execute_and(self, input1: np.ndarray, input2: np.ndarray) -> Optional[np.ndarray]:
        """Perform bitwise AND between the two input images."""
        try:
            # Ensure both images have the same dimensions
            if input1.shape != input2.shape:
                # Resize the second image to match the first
                input2 = cv2.resize(input2, (input1.shape[1], input1.shape[0]))
            
            # Ensure both images have the same number of channels
            if len(input1.shape) != len(input2.shape):
                if len(input1.shape) == 2 and len(input2.shape) == 3:
                    # Convert input2 to grayscale
                    input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY)
                elif len(input1.shape) == 3 and len(input2.shape) == 2:
                    # Convert input2 to BGR
                    input2 = cv2.cvtColor(input2, cv2.COLOR_GRAY2BGR)
            
            # Ensure both images have the same data type
            if input1.dtype != input2.dtype:
                input2 = input2.astype(input1.dtype)
            
            # Perform bitwise AND
            result = cv2.bitwise_and(input1, input2)
            return result
            
        except Exception as e:
            print(f"Error executing AND: {e}")
            return None
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """This method is not used for AndNode since it needs two inputs."""
        # AndNode uses _execute_and instead
        return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) == 2:  # Only execute if we have both inputs
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw AND icon (ampersand symbol)."""
        painter.save()
        try:
            # Dark blue color for AND nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 139), 2))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 139, 50)))
            
            # Draw ampersand symbol (&)
            center = icon_rect.center()
            size = 4
            
            # Draw the ampersand shape
            # Upper circle
            painter.drawEllipse(QtCore.QPointF(center.x(), center.y() - 1), size * 0.6, size * 0.6)
            # Lower circle
            painter.drawEllipse(QtCore.QPointF(center.x(), center.y() + 1), size * 0.6, size * 0.6)
            # Connecting line
            painter.drawLine(
                QtCore.QPointF(center.x() - size * 0.3, center.y() - 0.5),
                QtCore.QPointF(center.x() + size * 0.3, center.y() + 0.5)
            )
        finally:
            painter.restore()


class DiffNode(FunctionNode):
    """Node for calculating the difference between two images."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("Diff", icon_size, grid_size, {"name": "diff", "in": "Mat (BGR/Gray) - Mat (BGR/Gray)", "out": "Mat (BGR/Gray)"})
        self._parameters = {}
        # Initialize missing attributes from base class
        self._propagating = False
    
    def can_accept_input(self, input_node: Node) -> bool:
        """Check if this function can accept the given input."""
        # Diff node can accept up to 2 inputs
        return len(self._input_connections) < 2
    
    def _check_and_execute(self) -> None:
        """Override to handle two-input execution for DiffNode."""
        if self._executing or self._propagating:
            return
        
        # DiffNode needs exactly 2 inputs to execute
        if len(self._input_connections) != 2:
            return
        
        # Check if both inputs have valid images
        input1 = self._input_connections[0].get_output_image()
        input2 = self._input_connections[1].get_output_image()
        
        if input1 is None or input2 is None:
            return
        
        self._executing = True
        try:
            # Perform the difference calculation
            result = self._execute_diff(input1, input2)
            if result is not None:
                self._result_image = result
                self._update_result_thumbnail()
        finally:
            self._executing = False
    
    def _execute_diff(self, input1: np.ndarray, input2: np.ndarray) -> Optional[np.ndarray]:
        """Calculate the difference between the two input images."""
        try:
            # Ensure both images have the same dimensions
            if input1.shape != input2.shape:
                # Resize the second image to match the first
                input2 = cv2.resize(input2, (input1.shape[1], input1.shape[0]))
            
            # Ensure both images have the same number of channels
            if len(input1.shape) != len(input2.shape):
                if len(input1.shape) == 2 and len(input2.shape) == 3:
                    # Convert input2 to grayscale
                    input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY)
                elif len(input1.shape) == 3 and len(input2.shape) == 2:
                    # Convert input2 to BGR
                    input2 = cv2.cvtColor(input2, cv2.COLOR_GRAY2BGR)
            
            # Ensure both images have the same data type
            if input1.dtype != input2.dtype:
                input2 = input2.astype(input1.dtype)
            
            # Calculate the difference (input1 - input2) and clip to [0,255]
            #result = input1.astype(np.float32) - input2.astype(np.float32)
            #result = np.clip(result, 0, 255).astype(np.uint8)
            #result = cv2.absdiff(input1, input2)
            result = cv2.subtract(input1,input2)
            return result
            
        except Exception as e:
            print(f"Error executing diff: {e}")
            return None
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """This method is not used for DiffNode since it needs two inputs."""
        # DiffNode uses _execute_diff instead
        return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) == 2:  # Only execute if we have both inputs
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw diff icon (minus sign)."""
        painter.save()
        try:
            # Red color for diff nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(220, 20, 60), 2))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(220, 20, 60, 50)))
            
            # Draw minus sign
            center = icon_rect.center()
            size = 4
            
            # Horizontal line
            painter.drawLine(
                QtCore.QPointF(center.x() - size, center.y()),
                QtCore.QPointF(center.x() + size, center.y())
            )
        finally:
            painter.restore()


class MSERNode(FunctionNode):
    """Node for MSER (Maximally Stable Extremal Regions) detection."""
    
    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__("MSER", icon_size, grid_size, {"name": "mser", "in": "Mat (Gray)", "out": "Mat (BGR)"})
        self._parameters = {
            "delta": 5,
            "min_area": 60,
            "max_area": 14400,
            "max_variation": 0.25,
            "min_diversity": 0.2,
            "max_evolution": 200,
            "area_threshold": 1.01,
            "min_margin": 0.003,
            "edge_blur_size": 5
        }
        # Initialize missing attributes from base class
        self._propagating = False
    
    def _execute_function(self, input_image: np.ndarray) -> Optional[np.ndarray]:
        """Apply MSER detection to the input image."""
        try:
            # Convert to grayscale if needed
            if len(input_image.shape) == 3:
                gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_image.copy()
            
            # Create MSER detector with parameters
            params = self._parameters
            mser = cv2.MSER_create(
                delta=params["delta"],
                min_area=params["min_area"],
                max_area=params["max_area"],
                max_variation=params["max_variation"],
                min_diversity=params["min_diversity"],
                max_evolution=params["max_evolution"],
                area_threshold=params["area_threshold"],
                min_margin=params["min_margin"],
                edge_blur_size=params["edge_blur_size"]
            )
            
            # Detect regions
            regions, _ = mser.detectRegions(gray)
            
            # Create output image (convert to BGR for colored output)
            if len(input_image.shape) == 3:
                output = input_image.copy()
            else:
                output = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            
            # Fill each region with its mean color
            for region in regions:
                if len(region) < 3:  # Skip regions with too few points
                    continue
                
                # Convert region to proper format - MSER returns points as [x, y]
                region_points = np.array(region, dtype=np.int32)
                
                # Sample pixels from the region to calculate mean color
                sample_size = min(100, len(region_points))
                if sample_size < len(region_points):
                    # Randomly sample pixels
                    sample_indices = np.random.choice(len(region_points), sample_size, replace=False)
                    sample_pixels = region_points[sample_indices]
                else:
                    sample_pixels = region_points
                
                # Calculate mean color from sampled pixels
                sample_colors = []
                for pixel in sample_pixels:
                    x, y = pixel[0], pixel[1]
                    if 0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]:
                        sample_colors.append(input_image[y, x])
                
                if not sample_colors:
                    continue
                
                if len(input_image.shape) == 3:
                    # BGR image
                    sample_colors = np.array(sample_colors)
                    mean_color = np.mean(sample_colors, axis=0)
                else:
                    # Grayscale image
                    sample_colors = np.array(sample_colors)
                    mean_color = np.mean(sample_colors)
                    # Convert to BGR for output
                    mean_color = np.array([mean_color, mean_color, mean_color])
                
                mean_color = tuple(int(c) for c in mean_color)
                
                # Set all pixels in the region to the mean color
                for pixel in region_points:
                    x, y = pixel[0], pixel[1]
                    if 0 <= x < output.shape[1] and 0 <= y < output.shape[0]:
                        output[y, x] = mean_color
                   

            return output
            
        except Exception as e:
            print(f"Error executing MSER: {e}")
            return None
    
    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        """Set a parameter value."""
        if param_name in self._parameters:
            self._parameters[param_name] = value
            if len(self._input_connections) > 0:
                if preview_mode:
                    self._re_execute_preview()
                else:
                    self._check_and_execute()
                    # When not in preview mode, propagate changes to downstream functions
                    # but only if we're not already propagating to prevent infinite loops
                    if not getattr(self, '_propagating', False):
                        self._propagate_changes()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return self._parameters.copy()
    
    def _draw_specific_type_icon(self, painter: QtGui.QPainter, icon_rect: QtCore.QRectF) -> None:
        """Draw MSER icon (irregular shapes)."""
        painter.save()
        try:
            # Green color for MSER nodes
            painter.setPen(QtGui.QPen(QtGui.QColor(34, 139, 34), 2))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(34, 139, 34, 50)))
            
            # Draw irregular shapes to represent regions
            center = icon_rect.center()
            
            # Draw multiple irregular shapes
            for i in range(3):
                # Create a small irregular polygon
                points = []
                for j in range(4):
                    angle = (i * 120 + j * 90) * np.pi / 180
                    radius = 2 + i * 0.5
                    x = center.x() + radius * np.cos(angle)
                    y = center.y() + radius * np.sin(angle)
                    points.append(QtCore.QPointF(x, y))
                
                # Draw the polygon
                polygon = QtGui.QPolygonF(points)
                painter.drawPolygon(polygon)
        finally:
            painter.restore()
