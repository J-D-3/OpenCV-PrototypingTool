"""Standalone window for inspecting a node's output image (frontend)."""
from PyQt6 import QtCore, QtGui, QtWidgets

from ui.nodes import Node
from ui.image_utils import cv_to_qimage

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


