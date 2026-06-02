"""Small per-operation type glyphs drawn into the node's corner badge (frontend).

Hand-drawn with QPainter (no font asset / licensing) and keyed by operation id,
so the backend stays free of any icon concerns. Each drawer paints inside the
~16px badge rect, tinted with the operation's color.
"""
from PyQt6 import QtCore, QtGui

# operation id -> glyph key
ICON_BY_OP = {
    "save_to_file": "save",
    "create_batch": "batch",
    "to_grayscale": "convert",
    "to_bgr": "convert",
    "to_hls": "convert",
    "resize": "resize",
    "rotate": "rotate",
    "blur": "blur",
    "gaussian_blur": "blur",
    "threshold": "threshold",
    "adaptive_threshold": "threshold",
    "morphology": "morph",
    "normalize": "levels",
    "invert": "invert",
    "local_hdr": "sun",
    "canny": "edges",
    "sobel": "edges",
    "laplacian": "edges",
    "mser": "cluster",
    "sum": "plus",
    "and": "amp",
    "diff": "minus",
    "histogram": "histogram",
    "kmeans": "cluster",
    "auto_cluster": "cluster",
    "mean_shift": "cluster",
    "reduce_colors": "palette",
    "find_contours": "contours",
    "contour_filter": "contours",
    "dft": "fourier",
    "idft": "fourier",
}

_NoBrush = QtCore.Qt.BrushStyle.NoBrush
_AlignCenter = QtCore.Qt.AlignmentFlag.AlignCenter


def _inset(r: QtCore.QRectF, f: float = 0.12) -> QtCore.QRectF:
    return r.adjusted(r.width() * f, r.height() * f, -r.width() * f, -r.height() * f)


def _pen(p, c, w=1.4):
    pen = QtGui.QPen(c, w)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(_NoBrush)


def _glyph_text(p, r, c, ch):
    _pen(p, c)
    f = p.font()
    f.setBold(True)
    f.setPixelSize(max(7, int(r.height() * 0.95)))
    p.setFont(f)
    p.drawText(r, _AlignCenter, ch)


def _generic(p, r, c):
    p.setBrush(c)
    _pen(p, c)
    p.drawEllipse(r.center(), r.width() * 0.16, r.width() * 0.16)


def _save(p, r, c):
    b = _inset(r, 0.14)
    _pen(p, c)
    p.drawRoundedRect(b, 1.5, 1.5)
    # top write-protect notch
    nw = b.width() * 0.28
    p.fillRect(QtCore.QRectF(b.right() - nw - b.width() * 0.18, b.top(), nw, b.height() * 0.32), c)
    # label slot
    p.drawRect(QtCore.QRectF(b.left() + b.width() * 0.22, b.bottom() - b.height() * 0.42,
                             b.width() * 0.56, b.height() * 0.3))


def _image(p, r, c):
    b = _inset(r, 0.14)
    _pen(p, c)
    p.drawRect(b)
    # sun
    p.setBrush(c)
    p.drawEllipse(QtCore.QPointF(b.left() + b.width() * 0.3, b.top() + b.height() * 0.3), 1.3, 1.3)
    # mountain
    p.setBrush(_NoBrush)
    poly = QtGui.QPolygonF([
        QtCore.QPointF(b.left(), b.bottom()),
        QtCore.QPointF(b.left() + b.width() * 0.4, b.top() + b.height() * 0.5),
        QtCore.QPointF(b.left() + b.width() * 0.65, b.bottom() - b.height() * 0.2),
        QtCore.QPointF(b.right(), b.bottom()),
    ])
    p.drawPolyline(poly)


def _batch(p, r, c):
    b = _inset(r, 0.1)
    _pen(p, c)
    w, h = b.width() * 0.6, b.height() * 0.6
    for i, off in enumerate((0.0, 0.18, 0.36)):
        rect = QtCore.QRectF(b.left() + b.width() * off, b.top() + b.height() * (0.36 - off),
                             w, h)
        p.setBrush(QtGui.QColor(255, 255, 255) if i < 2 else c)
        p.drawRect(rect)


def _convert(p, r, c):
    b = _inset(r, 0.1)
    p.setBrush(c)
    _pen(p, c)
    # half-filled circle = "recolor"
    path = QtGui.QPainterPath()
    path.moveTo(b.center())
    path.arcTo(b, 90, 180)
    path.closeSubpath()
    p.drawPath(path)
    p.setBrush(_NoBrush)
    p.drawEllipse(b)


def _resize(p, r, c):
    b = _inset(r, 0.12)
    _pen(p, c)
    p.drawRect(b)
    inner = QtCore.QRectF(b.left() + b.width() * 0.3, b.top() + b.height() * 0.3,
                          b.width() * 0.4, b.height() * 0.4)
    p.drawRect(inner)
    p.drawLine(b.topLeft(), inner.topLeft())


def _blur(p, r, c):
    _pen(p, c)
    ctr = r.center()
    for rad in (r.width() * 0.12, r.width() * 0.22, r.width() * 0.32):
        p.drawEllipse(ctr, rad, rad)


def _threshold(p, r, c):
    b = _inset(r, 0.16)
    _pen(p, c)
    p.drawRect(b)
    p.fillRect(QtCore.QRectF(b.left(), b.top(), b.width() / 2, b.height()), c)


def _morph(p, r, c):
    b = _inset(r, 0.18)
    p.setBrush(c)
    _pen(p, c, 1.0)
    s = b.width() / 3.0
    for rr in range(3):
        for cc in range(3):
            if (rr + cc) % 2 == 0:
                p.drawRect(QtCore.QRectF(b.left() + cc * s, b.top() + rr * s, s * 0.7, s * 0.7))


def _edges(p, r, c):
    b = _inset(r, 0.16)
    _pen(p, c)
    poly = QtGui.QPolygonF([
        QtCore.QPointF(b.left(), b.bottom()),
        QtCore.QPointF(b.left() + b.width() * 0.35, b.bottom()),
        QtCore.QPointF(b.left() + b.width() * 0.35, b.top() + b.height() * 0.35),
        QtCore.QPointF(b.left() + b.width() * 0.7, b.top() + b.height() * 0.35),
        QtCore.QPointF(b.left() + b.width() * 0.7, b.top()),
        QtCore.QPointF(b.right(), b.top()),
    ])
    p.drawPolyline(poly)


def _contours(p, r, c):
    b = _inset(r, 0.14)
    _pen(p, c)
    path = QtGui.QPainterPath()
    path.moveTo(b.left() + b.width() * 0.5, b.top())
    path.cubicTo(b.right(), b.top(), b.right(), b.bottom(), b.center().x(), b.bottom())
    path.cubicTo(b.left(), b.bottom(), b.left(), b.top(), b.left() + b.width() * 0.5, b.top())
    p.drawPath(path)


def _cluster(p, r, c):
    b = _inset(r, 0.14)
    p.setBrush(c)
    _pen(p, c, 1.0)
    pts = [(0.25, 0.3), (0.7, 0.25), (0.5, 0.72)]
    for fx, fy in pts:
        p.drawEllipse(QtCore.QPointF(b.left() + b.width() * fx, b.top() + b.height() * fy), 1.4, 1.4)


def _palette(p, r, c):
    b = _inset(r, 0.16)
    _pen(p, c, 1.0)
    shades = (c.lighter(150), c, c.darker(150))
    w = b.width() / 3.0
    for i, col in enumerate(shades):
        p.setBrush(col)
        p.drawRect(QtCore.QRectF(b.left() + i * w, b.top(), w * 0.92, b.height()))


def _histogram(p, r, c):
    b = _inset(r, 0.16)
    p.setBrush(c)
    _pen(p, c, 1.0)
    heights = (0.4, 0.85, 0.6, 1.0)
    w = b.width() / len(heights)
    for i, hf in enumerate(heights):
        bar_h = b.height() * hf
        p.drawRect(QtCore.QRectF(b.left() + i * w, b.bottom() - bar_h, w * 0.7, bar_h))


def _sun(p, r, c):
    import math
    b = _inset(r, 0.24)
    ctr = b.center()
    rad = b.width() * 0.30
    _pen(p, c, 1.2)
    p.setBrush(c)
    p.drawEllipse(ctr, rad, rad)
    p.setBrush(_NoBrush)
    for k in range(8):
        a = math.radians(k * 45)
        p.drawLine(QtCore.QPointF(ctr.x() + math.cos(a) * rad * 1.45,
                                  ctr.y() + math.sin(a) * rad * 1.45),
                   QtCore.QPointF(ctr.x() + math.cos(a) * rad * 2.1,
                                  ctr.y() + math.sin(a) * rad * 2.1))


def _invert(p, r, c):
    # square with one diagonal half filled = photographic negative
    b = _inset(r, 0.16)
    _pen(p, c, 1.0)
    p.drawRect(b)
    p.setBrush(c)
    p.drawPolygon(QtGui.QPolygonF([b.topLeft(), b.topRight(), b.bottomLeft()]))


def _levels(p, r, c):
    # A black->white tone ramp (filled triangle) = contrast normalization.
    b = _inset(r, 0.16)
    _pen(p, c, 1.0)
    p.setBrush(c)
    p.drawPolygon(QtGui.QPolygonF([b.bottomLeft(), b.bottomRight(), b.topRight()]))


def _rotate(p, r, c):
    import math
    b = _inset(r, 0.16)
    _pen(p, c)
    p.drawArc(b, 40 * 16, 280 * 16)   # most of a circle (gap where the arrowhead goes)
    cx, cy = b.center().x(), b.center().y()
    rx, ry = b.width() / 2.0, b.height() / 2.0
    a = math.radians(40)              # arc end point (0deg = 3 o'clock, CCW)
    ex, ey = cx + rx * math.cos(a), cy - ry * math.sin(a)
    p.drawLine(QtCore.QPointF(ex, ey), QtCore.QPointF(ex - rx * 0.45, ey - ry * 0.1))
    p.drawLine(QtCore.QPointF(ex, ey), QtCore.QPointF(ex + rx * 0.1, ey - ry * 0.5))


def _fourier(p, r, c):
    import math
    b = _inset(r, 0.12)
    _pen(p, c)
    path = QtGui.QPainterPath()
    n = 16
    for i in range(n + 1):
        x = b.left() + b.width() * i / n
        y = b.center().y() - math.sin(i / n * 2 * math.pi) * b.height() * 0.38
        path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
    p.drawPath(path)


_DRAWERS = {
    "generic": _generic, "save": _save, "image": _image, "batch": _batch,
    "convert": _convert, "resize": _resize, "rotate": _rotate, "blur": _blur, "threshold": _threshold,
    "morph": _morph, "levels": _levels, "invert": _invert, "sun": _sun, "edges": _edges,
    "contours": _contours, "cluster": _cluster,
    "palette": _palette, "histogram": _histogram, "fourier": _fourier,
    "plus": lambda p, r, c: _glyph_text(p, r, c, "+"),
    "amp": lambda p, r, c: _glyph_text(p, r, c, "&"),
    "minus": lambda p, r, c: _glyph_text(p, r, c, "−"),
}


def draw_key(painter: QtGui.QPainter, rect: QtCore.QRectF, key: str, color: QtGui.QColor) -> None:
    painter.save()
    try:
        _DRAWERS.get(key, _generic)(painter, rect, color)
    finally:
        painter.restore()


def draw(painter: QtGui.QPainter, rect: QtCore.QRectF, op_id: str, color: QtGui.QColor) -> None:
    draw_key(painter, rect, ICON_BY_OP.get(op_id, "generic"), color)
