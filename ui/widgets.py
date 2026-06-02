"""Shared small Qt widgets (frontend)."""
from PyQt6 import QtCore, QtGui, QtWidgets


class _LineHandle(QtWidgets.QSplitterHandle):
    """Splitter handle that paints a thin 1px line but keeps a wide grab area."""

    def paintEvent(self, _e):
        p = QtGui.QPainter(self)
        p.setPen(QtGui.QPen(QtGui.QColor(0x80, 0x80, 0x80)))
        r = self.rect()
        if self.orientation() == QtCore.Qt.Orientation.Horizontal:
            x = r.center().x()
            p.drawLine(x, r.top(), x, r.bottom())   # vertical divider
        else:
            y = r.center().y()
            p.drawLine(r.left(), y, r.right(), y)   # horizontal divider


class LineSplitter(QtWidgets.QSplitter):
    """QSplitter whose handles show a 1px line but are easy to grab (wide hit area)."""

    def __init__(self, orientation, parent=None, handle_width: int = 9):
        super().__init__(orientation, parent)
        self.setHandleWidth(handle_width)

    def createHandle(self):
        return _LineHandle(self.orientation(), self)
