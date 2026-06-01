"""Image <-> Qt conversion helpers (frontend)."""
import cv2
from PyQt6 import QtGui

def cv_to_qimage(image_bgr) -> QtGui.QImage:
    """Convert OpenCV BGR image to QImage."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channel = image_rgb.shape
    bytes_per_line = channel * width
    qimage = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    return qimage.copy()
