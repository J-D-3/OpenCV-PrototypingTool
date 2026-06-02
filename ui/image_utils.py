"""Image <-> Qt conversion helpers (frontend)."""
import cv2
import numpy as np
from PyQt6 import QtGui


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to displayable uint8.

    uint8 passes through; everything else (float magnitudes, int16/32, etc.) is
    min-max normalized to 0..255 so results like Fourier magnitudes or distance
    transforms are visible instead of clipped to noise.
    """
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    lo = float(np.nanmin(a)) if a.size else 0.0
    hi = float(np.nanmax(a)) if a.size else 0.0
    if hi > lo:
        a = (a - lo) / (hi - lo) * 255.0
    else:
        a = np.zeros_like(a)
    return a.astype(np.uint8)


def downscale_max(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink an image so its longest side is <= ``max_side`` (keeps aspect; no-op
    if already small enough). Cheap C resize, so a thumbnail never has to build a
    full-resolution QImage from a huge source."""
    if image is None or max_side <= 0:
        return image
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return image
    s = max_side / float(m)
    return cv2.resize(image, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def cv_to_qimage(image) -> QtGui.QImage:
    """Convert an OpenCV image (any dtype / 1,3,4 channels) to a QImage.

    Handles grayscale/binary (single channel), BGR, and BGRA, and normalizes
    non-8-bit data for display. Returns a null QImage for unsupported input.
    """
    if image is None:
        return QtGui.QImage()

    arr = to_uint8(image)

    if arr.ndim == 2:
        h, w = arr.shape
        arr = np.ascontiguousarray(arr)
        return QtGui.QImage(arr.data, w, h, w,
                            QtGui.QImage.Format.Format_Grayscale8).copy()

    if arr.ndim == 3:
        h, w, ch = arr.shape
        if ch == 1:
            return cv_to_qimage(arr[:, :, 0])
        if ch == 3:
            rgb = np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            return QtGui.QImage(rgb.data, w, h, 3 * w,
                                QtGui.QImage.Format.Format_RGB888).copy()
        if ch == 4:
            rgba = np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA))
            return QtGui.QImage(rgba.data, w, h, 4 * w,
                                QtGui.QImage.Format.Format_RGBA8888).copy()

    return QtGui.QImage()
