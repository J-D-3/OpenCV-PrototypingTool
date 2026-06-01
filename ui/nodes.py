"""Qt graphics items for graph nodes (frontend view layer).

One generic FunctionNode is driven by a core.operations.Operation; ImageNode
is a source. A later phase moves graph topology into core.graph so these
become pure observers of the model.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from PyQt6 import QtCore, QtGui, QtWidgets

from core.operations import REGISTRY
from core.batch import Batch
from ui.image_utils import cv_to_qimage

if TYPE_CHECKING:
    from ui.arrow import ArrowItem


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
        # Backend links (set by the GraphController when the node is registered).
        self.gnode = None
        self.controller = None

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
        
        # Draw an error border if the backend recorded a failure for this node.
        gnode = getattr(self, 'gnode', None)
        if gnode is not None and getattr(gnode, 'error', None):
            painter.save()
            try:
                pen = QtGui.QPen(QtGui.QColor(220, 20, 60))  # crimson = error
                pen.setWidth(3)
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

    def refresh_from_model(self) -> None:
        """Update the view from the backend node's result. Override as needed."""
        pass

    def on_commit(self) -> None:
        """Hook fired after a committed (non-preview) recompute. Override as needed."""
        pass

    def get_preview_image(self):
        """Image to show in the inspector. Defaults to the node's output."""
        return self.get_output_image()

    def get_summary(self) -> Dict[str, Any]:
        """Key facts to show in the inspector (e.g. {'contours': 42})."""
        return {}

    # --- batch element resolution -----------------------------------------
    def _cur_index(self, value) -> int:
        """Current preview index clamped to a batched value's length."""
        if isinstance(value, Batch) and value.items:
            idx = self.controller.preview_index if self.controller is not None else 0
            return max(0, min(idx, len(value.items) - 1))
        return 0

    def _element(self, value):
        """Resolve the currently-previewed element of a (possibly batched) value."""
        if isinstance(value, Batch):
            return value.items[self._cur_index(value)] if value.items else None
        return value


class ImageNode(Node):
    """Node representing an input image."""
    
    def __init__(self, source, icon_size: int, grid_size: int = 12):
        # ``source`` is a single image or a Batch of images.
        self._source = source
        first = source.items[0] if isinstance(source, Batch) else source
        h, w = first.shape[:2]
        channels = 1 if first.ndim == 2 else first.shape[2]
        dtype = first.dtype
        if channels == 1:
            kind = "Float" if np.issubdtype(dtype, np.floating) else "Gray"
        else:
            kind = "Float" if np.issubdtype(dtype, np.floating) else "BGR"

        meta = {"type": kind, "w": w, "h": h, "channels": channels}
        if isinstance(source, Batch):
            meta["count"] = len(source)
        super().__init__(icon_size, grid_size, meta)
        self._render_icon()

    def _render_icon(self) -> None:
        """Render the current image (or current batch element) as a thumbnail."""
        image = self._element(self._source)
        size = max(1, int(self._icon_size))
        thumb = QtGui.QPixmap(size, size)
        thumb.fill(QtGui.QColor(255, 255, 255))  # white background
        painter = QtGui.QPainter(thumb)
        try:
            if image is not None:
                pix = QtGui.QPixmap.fromImage(cv_to_qimage(image))
                scaled = pix.scaled(
                    size, size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                x = (size - scaled.width()) // 2
                y = (size - scaled.height()) // 2
                painter.drawPixmap(x, y, scaled)
        finally:
            painter.end()
        self.setPixmap(thumb)

    def refresh_from_model(self) -> None:
        # Re-render when the previewed batch element changes.
        self._render_icon()
        self._notify_arrows()

    def get_output_image(self) -> Optional[np.ndarray]:
        """Return the source image (current element when batched)."""
        return self._element(self._source)
    
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
    """View item for a registered operation.

    Topology, parameter values, and results live in the backend (``core.graph``
    via the :class:`~ui.controller.GraphController`). This class is a thin view:
    it delegates data operations to the controller and renders whatever result
    the backend currently holds for its node.
    """

    def __init__(self, op, icon_size: int, grid_size: int = 12):
        self.op = op
        self._label = op.label
        meta = {"name": op.id, "in": op.in_label, "out": op.out_label}
        super().__init__(icon_size, grid_size, meta)

    # --- rendering ---------------------------------------------------------
    def set_icon_size(self, icon_size: int) -> None:
        """Update the icon size and re-render (result thumbnail if available)."""
        self._icon_size = int(icon_size)
        self._render_current()
        self._notify_arrows()

    def _render_current(self) -> None:
        """Render the node's thumbnail from its display image, or the op icon.

        The display image is the op's preview (e.g. a cluster swatch), which may
        be absent or non-image for ops whose output is not itself an image — in
        that case we just draw the operation icon.
        """
        display = self.get_preview_image()
        if isinstance(display, np.ndarray):
            self._update_result_thumbnail(display)
        else:
            self._render_icon()

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

    def refresh_from_model(self) -> None:
        """Re-render from the backend result (called by the controller)."""
        self._render_current()
        self._notify_arrows()
        self.update()

    def _update_result_thumbnail(self, image) -> None:
        """Show the given result image as the node thumbnail."""
        if image is None:
            return

        qimage = cv_to_qimage(image)
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

    # --- data: delegate to the backend via the controller ------------------
    def get_output_image(self) -> Optional[np.ndarray]:
        return None if self.gnode is None else self._element(self.gnode.output)

    def get_preview_image(self):
        """Inspector image: the op's render_preview (e.g. contours drawn onto the
        input) if it defines one, otherwise the raw output. Batch-aware: resolves
        the currently-previewed element of the output and of each input."""
        out = self.get_output_image()
        render = getattr(self.op, "render_preview", None)
        if render is not None and self.gnode is not None and self.controller is not None:
            inputs = [n.get_output_image() if isinstance(n, Node) else None
                      for n in self._input_qt_nodes()]
            try:
                preview = render(inputs, out, dict(self.gnode.params))
                if preview is not None:
                    return preview
            except Exception as e:  # noqa: BLE001
                print(f"render_preview failed for {self.op.id}: {e}")
        return out

    def _input_qt_nodes(self):
        """The Qt nodes feeding this one (so previews resolve the same element)."""
        if self.controller is None or self.gnode is None:
            return []
        return [self.controller._qt_by_gid.get(src.id)
                for src in self.controller.model.inputs_of(self.gnode)]

    def get_summary(self) -> Dict[str, Any]:
        """Key facts from the op's summary hook (e.g. {'contours': 42})."""
        summarize = getattr(self.op, "summary", None)
        if summarize is None or self.gnode is None:
            return {}
        try:
            return summarize(self.get_output_image(), dict(self.gnode.params)) or {}
        except Exception as e:  # noqa: BLE001
            print(f"summary failed for {self.op.id}: {e}")
            return {}

    def can_accept_input(self, input_node: Node) -> bool:
        if self.controller is None:
            return False
        return self.controller.can_connect(input_node, self)

    def add_input_connection(self, input_node: Node) -> bool:
        if self.controller is None:
            return False
        return self.controller.connect(input_node, self)

    def set_parameter(self, param_name: str, value: Any, preview_mode: bool = False) -> None:
        if self.gnode is None or param_name not in self.gnode.params:
            return
        self.controller.set_param(self, param_name, value, commit=not preview_mode)

    def get_parameters(self) -> Dict[str, Any]:
        return {} if self.gnode is None else dict(self.gnode.params)


class SaveToFileNode(FunctionNode):
    """Saves its (passed-through) input image to ./output on a committed eval.

    The registry entry's compute is a pass-through, so the backend result is
    just the input image; the disk write is a view-layer side effect fired by
    ``on_commit`` (never during a parameter preview).
    """

    def __init__(self, icon_size: int, grid_size: int = 12):
        super().__init__(REGISTRY["save_to_file"], icon_size, grid_size)

    def on_commit(self) -> None:
        # Write the whole batch (one file per element), or the single image.
        out = self.gnode.output if self.gnode is not None else None
        if isinstance(out, Batch):
            for i, image in enumerate(out.items):
                if image is not None:
                    self._write_to_disk(image, suffix=f"_{i:03d}")
        elif out is not None:
            self._write_to_disk(out)

    def _write_to_disk(self, image, suffix: str = "") -> None:
        import os
        import time
        try:
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)

            if not hasattr(self, '_process_start_time'):
                self._process_start_time = time.time()
            if not hasattr(self, '_node_index'):
                self._node_index = id(self) % 10000

            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._process_start_time))
            default_filename = f"save_to_file_{ts}_{self._node_index}{suffix}.png"

            params = self.get_parameters()
            filename = params.get("filename", "")
            use_custom = params.get("use_custom", False)
            if not use_custom or not filename:
                filename = default_filename
            elif suffix:
                # Insert the batch suffix before the extension for custom names.
                stem, dot, ext = filename.rpartition(".")
                filename = f"{stem}{suffix}{dot}{ext}" if dot else f"{filename}{suffix}"

            if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
                filename += '.png'

            filepath = os.path.join(output_dir, filename)
            if cv2.imwrite(filepath, image):
                print(f"Image saved to: {filepath}")
            else:
                print(f"Failed to save image to: {filepath}")
        except Exception as e:
            print(f"Error saving image: {e}")
