"""Arrow graphics item connecting two nodes (frontend)."""
from PyQt6 import QtCore, QtGui, QtWidgets

from ui.nodes import Node

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


