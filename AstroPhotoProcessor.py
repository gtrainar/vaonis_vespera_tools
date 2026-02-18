##############################################
# AstroPhoto Processor
#
# (c) G. Trainar 2026
# SPDX-License-Identifier: MIT License
##############################################

# Version 1.0.0

"""
Astrophotography Processor Tool
Turn a hazy sky into a clean canvas in a few clicks.

This script uses wavelet decomposition and advanced filtering
to harmonize the sky background while preserving astronomical details.
"""

# ------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------
import sys, os, time, traceback, threading, shlex
from typing import Optional, Tuple, Any

import numpy as np

# ------------------------------------------------------------------
# Third‑party imports – may be missing at first run
# ------------------------------------------------------------------
try:
    import sirilpy as s
    from sirilpy import LogColor
except Exception:
    s = None

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox,
    QGroupBox, QWidget,
    QFileDialog, QMessageBox, QSplitter, QSizePolicy,
    QProgressBar, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter

import cv2
from skimage import color, filters
from scipy import signal

# ------------------------------------------------------------------
# CUSTOM WIDGETS
# ------------------------------------------------------------------
class OverlayGraphicsView(QWidget):
    """A widget that combines a scroll area with an overlay for progress bar."""
    
    def __init__(self, scroll_area, progress_bar):
        super().__init__()
        self.scroll_area = scroll_area
        self.progress_bar = progress_bar
        
        # Main layout with just the scroll area
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)
        
        # Add progress bar as overlay (absolute positioning)
        self.progress_bar.setParent(self)
        self.progress_bar.move(0, 0)
    
    def updateProgressBarPosition(self):
        """Update progress bar position to bottom of widget."""
        if self.progress_bar.isVisible():
            width = self.width()
            height = 10
            y_pos = self.height() - height
            self.progress_bar.setGeometry(0, y_pos, width, height)
    
    def resizeEvent(self, event):
        """Update progress bar position when widget is resized."""
        self.updateProgressBarPosition()
        super().resizeEvent(event)


# ------------------------------------------------------------------
# DEPENDENCY CHECK
# ------------------------------------------------------------------
def ensure_dependencies():
    """Install the required third‑party packages if they are missing."""
    try:
        import sirilpy as s
        deps = [
            "numpy",
            "scipy",
            "scikit-image",
            "opencv-python",
            "PyQt6",
            "pywavelets",
        ]
        for dep in deps:
            s.ensure_installed(dep)
        return True
    except Exception as e:
        print(f"Failed to ensure dependencies: {e}")
        return False


# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
VERSION = "1.0.0"
SCRIPT_NAME = "AstroPhoto Processor"

DEFAULT_PARAMS = {
    'wavelet_levels': 7,
    'median_kernel_size': 3,
    'sigma_threshold': 5.0,
    'denoise_threshold': 0.0,
    'dark_mask_opacity': 0.0,
    'preview_scale': 0.5, 
    'intensity': 0.4, 
}


# ------------------------------------------------------------------
# ENGINE
# ------------------------------------------------------------------
class AstroPhotoProcessorEngine:
    """Core processing engine for AstroPhoto Processor"""

    def __init__(self):
        self.params = DEFAULT_PARAMS.copy()
        self.last_processed: Optional[np.ndarray] = None
        self.processing_time: float = 0.0

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def process_image(self, image_data: np.ndarray,
                  zoom_factor: float = 1.0) -> np.ndarray:
        """Main processing pipeline (zoom_factor only used for smoothing)."""
        start_time = time.time()
        try:
            if image_data.dtype != np.float32:
                image_data = image_data.astype(np.float32)

            # 1. Background Intensity (now first)
            intensity = self.params.get('intensity', 0.0)

            # Convert to Lab color space
            lab_image = self._rgb_to_lab(image_data)
            luminance = lab_image[:, :, 0]
            original_shape = luminance.shape

            # 2. Outlier Rejection (moved up)
            background = self._estimate_background(luminance)

            # 3. Background Contrast (moved down)
            processed_luminance = luminance - background

            # 4. Wavelet Decomposition (moved down)
            wavelet_scales = self._wavelet_decompose(processed_luminance)
            processed_scales = self._process_background(wavelet_scales)

            # 5. Noise Reduction (moved down)
            processed_luminance = self._wavelet_reconstruct(processed_scales, original_shape)

            # 6. Dark Mask (moved to bottom)
            if self.params['dark_mask_opacity'] > 0:
                # Apply dark mask to luminance channel
                processed_luminance = self._apply_dark_mask(processed_luminance)

            # Combine back to Lab and then RGB
            result_lab = np.stack(
                [processed_luminance,
                lab_image[:, :, 1],
                lab_image[:, :, 2]],
                axis=2)

            result_rgb = self._lab_to_rgb(result_lab)

            # Apply intensity blending (now using the processed result)
            if intensity != 1.0:
                result_rgb = (1 - intensity) * image_data + intensity * result_rgb

            # Final clipping
            result_rgb = np.clip(result_rgb, 0, 1)
            self.last_processed = result_rgb
            self.processing_time = time.time() - start_time
            return result_rgb

        except Exception as e:
            raise RuntimeError(f"Processing failed: {e}")



    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _rgb_to_lab(self, rgb_image: np.ndarray) -> np.ndarray:
        return color.rgb2lab(rgb_image)

    def _lab_to_rgb(self, lab_image: np.ndarray) -> np.ndarray:
        return color.lab2rgb(lab_image)

    def _wavelet_decompose(self, image: np.ndarray) -> list:
        try:
            import pywt
            h, w = image.shape[:2]
            max_level = self.params['wavelet_levels']
            max_possible_level = min((h.bit_length() - 1), (w.bit_length() - 1))
            actual_level = min(max_level, max_possible_level)
            if actual_level < max_level:
                print(f"WARNING: Reduced wavelet levels from {max_level} to {actual_level} due to image size {h}x{w}")
            coeffs = pywt.wavedec2(image, 'db4', level=actual_level)
            return coeffs
        except Exception:
            return [image]

    def _wavelet_reconstruct(self, scales: list, target_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct the image from wavelet coefficients and upsample
        to the original shape."""
        try:
            import pywt
            approx = pywt.waverec2(scales, 'db4')
        except Exception as e:
            raise RuntimeError(f"Wavelet reconstruction failed: {e}")

        if approx.shape != target_shape:
            from skimage.transform import resize
            approx = resize(approx, target_shape,
                            order=1, preserve_range=True)
        return approx

    def _process_background(self, scales: list) -> list:
        try:
            import pywt
            approx = scales[0]
            detail_levels = scales[1:]

            background = self._estimate_background(approx)
            processed_approx = approx - background

            processed_details = []
            for details in detail_levels:
                if isinstance(details, tuple) and len(details) == 3:
                    h_processed = self._denoise_scale(details[0])
                    v_processed = self._denoise_scale(details[1])
                    d_processed = self._denoise_scale(details[2])
                    processed_details.append((h_processed, v_processed, d_processed))
                else:
                    processed_details.append(details)

            return [processed_approx] + processed_details

        except Exception:
            return self._process_background_fallback(scales)

    def _process_background_fallback(self, scales: list) -> list:
        processed = []
        for level, scale in enumerate(scales):
            if level == len(scales) - 1:
                background = self._estimate_background(scale)
                processed.append(scale - background)
            else:
                processed.append(self._denoise_scale(scale))
        return processed

    def _estimate_background(self, approximation: np.ndarray) -> np.ndarray:
        kernel = self.params['median_kernel_size']
        if kernel < 3:
            kernel = 3
        if kernel % 2 == 0: 
            kernel += 1

        from skimage.morphology import rectangle
        footprint = rectangle(kernel, kernel)

        median_filtered = filters.median(approximation, footprint=footprint)
        blurred = filters.gaussian(median_filtered, sigma=1.0)

        std = np.std(blurred)
        threshold = self.params['sigma_threshold'] * std
        mask = np.abs(blurred - np.median(blurred)) < threshold

        background = blurred.copy()
        background[~mask] = np.median(blurred)
        return background

    def _denoise_scale(self, scale_data: np.ndarray) -> np.ndarray:
        threshold = self.params['denoise_threshold'] * np.std(scale_data)
        return np.sign(scale_data) * np.maximum(np.abs(scale_data) - threshold, 0)

    def _apply_dark_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply a radial darkening mask to the image (works with both RGB and single-channel)."""
        opacity = self.params['dark_mask_opacity']
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

        mask = 1.0 - (distance / max_dist)
        mask = 1.0 - (opacity * mask)

        if len(image.shape) == 2:  # Single channel
            return image * mask
        else:  # Multi-channel
            mask = mask[:, :, np.newaxis]
            return image * mask

    def _apply_edge_preserving_smoothing(
        self,
        image: np.ndarray,
        mod: float,
        zoom_factor: float = 1.0
    ) -> np.ndarray:
        """Edge‑preserving smoothing using a bilateral filter whose diameter adapts to zoom."""
        # Compute diameter = 3 × zoom_factor, keep it odd and at least 3
        d = int(round(3 * zoom_factor * 1.414))
        if d < 3:
            d = 3
        if d % 2 == 0:          # cv2.bilateralFilter expects an odd diameter
            d += 1

        img_float = image.astype(np.float32)
        smoothed = cv2.bilateralFilter(
            img_float,
            d=d,                # dynamic diameter
            sigmaColor=11.0,
            sigmaSpace=11.0
        )
        return (1 - mod) * image + mod * smoothed


# ------------------------------------------------------------------
# PREVIEW WORKER
# ------------------------------------------------------------------
class PreviewWorker(QThread):
    preview_ready = pyqtSignal(object)
    processing_time = pyqtSignal(float)

    def __init__(self, engine: AstroPhotoProcessorEngine, preview_label):
        super().__init__()
        self.engine = engine
        self.preview_label = preview_label  # new attribute
        self.full_image: Optional[np.ndarray] = None
        self._is_running = False
        self._lock = threading.Lock()

    def set_image(self, image_data: np.ndarray):
        with self._lock:
            self.full_image = image_data

    def stop(self):
        self._is_running = False
        self.wait()
        with self._lock:
            self.full_image = None

    def run(self):
        """Main preview loop"""
        self._is_running = True
        while self._is_running:
            start_time = time.time()
            with self._lock:
                if self.full_image is not None:
                    try:
                        preview_image = self._create_preview()
                        self.preview_ready.emit(preview_image)
                        self.processing_time.emit(self.engine.processing_time)
                    except Exception as e:
                        print(f"Preview error: {e}")
                        break
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed)
            time.sleep(sleep_time)

    def _create_preview(self):
        scale_factor = self.engine.params['preview_scale']
        small_img = cv2.resize(
            self.full_image,
            (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA)
        # Grab current zoom factor from the preview widget
        zoom = getattr(self.preview_label, 'zoom_factor', 1.0)
        return self.engine.process_image(small_img, zoom_factor=zoom)


# ------------------------------------------------------------------
# Zoomable preview widget
# ------------------------------------------------------------------
class ZoomableLabel(QLabel):
    """A QLabel that supports zooming, panning and fitting."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.base_image: Optional[QImage] = None
        self.zoom_factor: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0

        self.manual_zoom: bool = False
        self.is_panning: bool = False
        self.pan_start_pos: Optional[QPoint] = None

    def setBaseImage(self, qimage: QImage):
        self.base_image = qimage
        if self.zoom_factor == 0:
            self.zoom_factor = 1.0
        self.updatePixmap()

    def updatePixmap(self) -> None:
        """Render the image with the current zoom / pan settings."""
        if self.base_image is None:
            self.clear()
            return

        view_w, view_h = self.width(), self.height()
        img_w, img_h     = self.base_image.width(), self.base_image.height()

        # Only auto-fit if we're not in manual zoom mode
        if self.zoom_factor == 0 and not self.manual_zoom:
            desired_zoom = max(view_w / img_w, view_h / img_h)
            self.zoom_factor = desired_zoom
            self.offset_x = 0
            self.offset_y = 0

        # Calculate scaled dimensions
        w = int(img_w * self.zoom_factor)
        h = int(img_h * self.zoom_factor)

        if w <= 0 or h <= 0:
            return

        # Create scaled pixmap
        pixmap = QPixmap.fromImage(self.base_image).scaled(
            w, h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

        # Center the image if it's smaller than the view
        if w < view_w or h < view_h:
            x = (view_w - w) // 2
            y = (view_h - h) // 2
            self.setPixmap(pixmap)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            # For larger images, use the offset system
            max_x = w - view_w
            max_y = h - view_h
            self.offset_x = max(0, min(self.offset_x, max_x))
            self.offset_y = max(0, min(self.offset_y, max_y))

            # Create a transparent pixmap of the view size
            full_pixmap = QPixmap(view_w, view_h)
            full_pixmap.fill(Qt.GlobalColor.transparent)

            # Draw the cropped portion
            painter = QPainter(full_pixmap)
            source_rect = QRect(self.offset_x, self.offset_y, view_w, view_h)
            painter.drawPixmap(0, 0, pixmap.copy(source_rect))
            painter.end()

            self.setPixmap(full_pixmap)
            self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)


    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.offset_x -= delta.x()
            self.offset_y -= delta.y()
            self.pan_start_pos = event.pos()
            self.updatePixmap()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.2 if delta > 0 else 0.833333
        
        mouse_pos = event.position().toPoint()
        
        img_w, img_h = self.base_image.width(), self.base_image.height()
        view_w, view_h = self.width(), self.height()
        
        current_w = int(img_w * self.zoom_factor)
        current_h = int(img_h * self.zoom_factor)
        
        anchor_x = self.offset_x + mouse_pos.x()
        anchor_y = self.offset_y + mouse_pos.y()
        
        new_zoom = self.zoom_factor * factor
        self.zoom_factor = max(0.1, min(new_zoom, 10))
        
        new_w = int(img_w * self.zoom_factor)
        new_h = int(img_h * self.zoom_factor)
        
        ratio_x = anchor_x / current_w if current_w > 0 else 0
        ratio_y = anchor_y / current_h if current_h > 0 else 0
        
        new_anchor_x = int(ratio_x * new_w)
        new_anchor_y = int(ratio_y * new_h)
        
        self.offset_x = new_anchor_x - mouse_pos.x()
        self.offset_y = new_anchor_y - mouse_pos.y()
        
        max_x = max(0, new_w - view_w)
        max_y = max(0, new_h - view_h)
        self.offset_x = max(0, min(self.offset_x, max_x))
        self.offset_y = max(0, min(self.offset_y, max_y))
        
        self.updatePixmap()
        super().wheelEvent(event)

    def _on_fit_clicked(self):
        self.preview_label.zoom_factor = 0.0 
        self.preview_label.updatePixmap()

    def _adjust_offset_after_zoom(self, factor: float):
        """Scale by *factor* while keeping the viewport centre fixed."""
  
        parent = self.parent()
        if isinstance(parent, QScrollArea):
            view_w, view_h = parent.viewport().size().width(), parent.viewport().size().height()
        else:
            view_w, view_h = self.width(), self.height()

        img_w, img_h = self.base_image.width(), self.base_image.height()

        center_x = self.offset_x + view_w / 2.0
        center_y = self.offset_y + view_h / 2.0

        self.zoom_factor *= factor

        w = int(img_w * self.zoom_factor)
        h = int(img_h * self.zoom_factor)

        if w > view_w:
            self.offset_x = int(center_x - view_w / 2.0)
        else:
            self.offset_x = 0

        if h > view_h:
            self.offset_y = int(center_y - view_h / 2.0)
        else:
            self.offset_y = 0

        max_x, max_y = max(0, w - view_w), max(0, h - view_h)
        self.offset_x = max(0, min(self.offset_x, max_x))
        self.offset_y = max(0, min(self.offset_y, max_y))

    def zoomIn(self):
        self._zoom_with_center(1.2)
        self.manual_zoom = True

    def zoomOut(self):
        self._zoom_with_center(0.833333)
        self.manual_zoom = True

    def _zoom_with_center(self, factor: float):
        """Zoom around the center of the viewport."""
        if self.base_image is None:
            return
            
        view_w, view_h = self.width(), self.height()
        img_w, img_h = self.base_image.width(), self.base_image.height()
        
        center_x = view_w / 2.0
        center_y = view_h / 2.0
        
        current_w = int(img_w * self.zoom_factor)
        current_h = int(img_h * self.zoom_factor)
        
        image_center_x = self.offset_x + center_x
        image_center_y = self.offset_y + center_y
        
        new_zoom = self.zoom_factor * factor
        self.zoom_factor = max(0.1, min(new_zoom, 10))
        
        new_w = int(img_w * self.zoom_factor)
        new_h = int(img_h * self.zoom_factor)
        
        if current_w > 0 and current_h > 0:
            ratio_x = image_center_x / current_w
            ratio_y = image_center_y / current_h
            new_image_center_x = ratio_x * new_w
            new_image_center_y = ratio_y * new_h
        else:
            new_image_center_x = new_w / 2.0
            new_image_center_y = new_h / 2.0
        
        self.offset_x = round(new_image_center_x - center_x)
        self.offset_y = round(new_image_center_y - center_y)
        
        max_x = max(0, new_w - view_w)
        max_y = max(0, new_h - view_h)
        self.offset_x = max(0, min(self.offset_x, max_x))
        self.offset_y = max(0, min(self.offset_y, max_y))
        
        self.updatePixmap()

    def resetZoom(self):
        self.zoom_factor = 1.0
        self.manual_zoom = True

        if self.base_image is not None:
            img_w, img_h = self.base_image.width(), self.base_image.height()
            view_w, view_h = self.width(), self.height()

            scaled_w = int(img_w * self.zoom_factor)
            scaled_h = int(img_h * self.zoom_factor)

            if scaled_w > view_w:
                self.offset_x = (scaled_w - view_w) // 2
            else:
                self.offset_x = 0

            if scaled_h > view_h:
                self.offset_y = (scaled_h - view_h) // 2
            else:
                self.offset_y = 0
        else:
            self.offset_x = 0
            self.offset_y = 0

        self.updatePixmap()

    def fitToWindow(self):
        if self.base_image is None:
            return

        view_w, view_h = self.width(), self.height()
        img_w, img_h = self.base_image.width(), self.base_image.height()

        # Calculate zoom factors for width and height
        zoom_w = view_w / img_w
        zoom_h = view_h / img_h

        # Use the smaller zoom factor to fit entirely
        self.zoom_factor = min(zoom_w, zoom_h)
        self.manual_zoom = False
        # Reset offsets to show from top-left
        self.offset_x = 0
        self.offset_y = 0

        # Force update with the new zoom factor
        self.updatePixmap()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Only auto-fit if we're not in manual zoom mode
        if not self.manual_zoom:
            self.fitToWindow()


# ------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------
class AstroPhotoProcessorGUI(QDialog):
    """Main GUI for the AstroPhoto Processor."""

    def __init__(self, siril_interface: Any,
                 app: QApplication,
                 resize_delay_ms: int = 200):
        super().__init__()
        self.siril = siril_interface
        self.app    = app

        self.engine         = AstroPhotoProcessorEngine()
        self._build_ui()

        self.preview_worker = PreviewWorker(self.engine, self.preview_label)  # will be re‑created after UI build

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True) 
        self._preview_timer.timeout.connect(self._start_preview)

        self._dark_mask_timer = QTimer(self)
        self._dark_mask_timer.setSingleShot(True)  
        self._dark_mask_timer.timeout.connect(self._apply_dark_mask_preview)

        self._resize_delay_ms = resize_delay_ms
        self._resize_timer    = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(
            self._restart_preview_after_resize
        )

        self.original_image: Optional[np.ndarray] = None
        self.current_image:  Optional[np.ndarray] = None

        self._is_resizing = False
        self.show_original = False
        self._first_preview_done: bool = False
        self.original_file_path: Optional[str] = None       

        self._setup_connections()

        self.setStyleSheet(self._get_dark_stylesheet())

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _get_dark_stylesheet(self) -> str:
        return """
        QDialog { background-color: #2b2b2b; color: #e0e0e0; }
        QGroupBox { border: 1px solid #444444; margin-top: 12px;
                    font-weight: bold; border-radius: 4x; padding-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10x;
                           padding: 0 5px; color: #88aaff; }
        QLabel { color: #cccccc; font-size: 10pt; }
        QLabel#title { color: #88aaff; font-size: 14pt; font-weight: bold; }
        QSlider::groove:horizontal { border: 1px solid #555555; height: 8px;
                                     background: #3c3c3c; border-radius: 4x; }
        QSlider::handle:horizontal { background: #88aaff; border: 1px solid #555555;
                                     width: 16px; height: 16px; margin: -4px 0;
                                     border-radius: 8x; }
        QPushButton { background-color: #444444; color: #dddddd;
                      border: 1px solid #666666; border-radius: 4x;
                      padding: 8px 20px; font-weight: bold; min-width: 100px; }
        QPushButton:hover { background-color: #555555; }
        QPushButton:pressed { background-color: #333333; }
        QPushButton#process { background-color: #285299; }
        """

    def _build_ui(self) -> None:
        """Create the entire dialog layout – called from __init__."""
        # ------------------------------------------------------------------
        # Splitter: left (controls) | right (preview)
        # ------------------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setHandleWidth(4)

        # ------------------------------------------------------------------
        # Left side – control panel
        # ------------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Create a container that matches the preview group structure
        self.left_container = QGroupBox()
        left_container_layout = QVBoxLayout(self.left_container)
        left_container_layout.setContentsMargins(10, 10, 10, 10)

        # Add processing tab
        left_container_layout.addWidget(self._create_processing_tab())

        # Add help button (centered like zoom buttons)
        self.help_btn = QPushButton("Help")
        self.help_btn.clicked.connect(self._show_help)

        # Use a layout for the help button to center it
        help_layout = QHBoxLayout()
        help_layout.addStretch()
        help_layout.addWidget(self.help_btn)
        help_layout.addStretch()
        left_container_layout.addLayout(help_layout)

        # Add the container to left_widget
        left_layout.addWidget(self.left_container)

        # ------------------------------------------------------------------
        # Right side – preview
        # ------------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Preview group
        preview_group = QGroupBox()
        preview_layout = QVBoxLayout(preview_group)

        # Zoomable preview label
        self.preview_label = ZoomableLabel()
        self.preview_label.setMinimumSize(512, 512)
        self.preview_label.setStyleSheet(
            """
            background-color: #111;
            border: 1px solid #444;
            """
        )
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Progress bar (hidden by default) - will be added to overlay
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        # Set progress bar to be at the bottom of the overlay
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #333; color: white; border: 1px solid #555; text-align: center; } QProgressBar::chunk { background-color: #88aaff; }")
        self.progress_bar.setFixedHeight(10)

        # Create overlay widget for progress bar to prevent layout shifts
        self.overlay_widget = OverlayGraphicsView(self.scroll_area, self.progress_bar)
        preview_layout.addWidget(self.overlay_widget)

        # ------------------------------------------------------------------
        # ORIGINAL indicator
        # ------------------------------------------------------------------
        self.original_indicator = QLabel("ORIGINAL", self.preview_label)
        self.original_indicator.setStyleSheet("""
            background-color: rgba(0, 0, 0, 128);
            color: #ffffff;
            padding: 2px 4px;
            border-radius: 3px;
        """)
        self.original_indicator.move(10, 10)          
        self.original_indicator.setVisible(False)

        # Zoom / fit buttons
        zoom_btns = QHBoxLayout()
        self.zoom_out_btn = QPushButton("Zoom –")
        self.zoom_in_btn  = QPushButton("Zoom +")
        self.fit_btn      = QPushButton("Fit")
        self.one_to_one_btn = QPushButton("1:1")

        zoom_btns.addWidget(self.zoom_out_btn)
        zoom_btns.addWidget(self.zoom_in_btn)
        zoom_btns.addWidget(self.fit_btn)
        zoom_btns.addWidget(self.one_to_one_btn)

        preview_layout.addLayout(zoom_btns)

        right_layout.addWidget(preview_group)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)   
        splitter.setStretchFactor(1, 2)   

        # ------------------------------------------------------------------
        # Main dialog layout
        # ------------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel()
        header.setText(
            f"<div style='text-align:center;'>"
            f"<h1>{SCRIPT_NAME}</h1>"
            "<p style='font-size:10pt; color:#ff8800;'>"
            "Space bar toggles between original image and computed image"
            "</p>"
            "</div>")
        header.setObjectName("title")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFixedHeight(60)          
        main_layout.addWidget(header)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("status")
        self.status_label.setStyleSheet(
            "color: #ffcc00; font-size: 10pt;"
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(40)
        main_layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self._load_image)
        button_layout.addWidget(self.btn_load)

        self.btn_process = QPushButton("Process Full Image")
        self.btn_process.setObjectName("process")
        self.btn_process.clicked.connect(self._process_full_image)
        self.btn_process.setEnabled(False)
        button_layout.addWidget(self.btn_process)

        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self._save_result)
        self.btn_save.setEnabled(False)
        button_layout.addWidget(self.btn_save)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Add splitter
        main_layout.addWidget(splitter)


    # ------------------------------------------------------------------
    # Create processing tab (sliders, checkboxes)
    # ------------------------------------------------------------------
    def _create_processing_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # ------------------------------------------------------------------
        # Intensity slider 
        # ------------------------------------------------------------------
        intensity_group = QGroupBox("Background Intensity")
        intensity_layout = QVBoxLayout(intensity_group)

        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(
            int(self.engine.params['intensity'] * 100))
        self.intensity_slider.setTickInterval(10)

        intensity_control = QHBoxLayout()
        intensity_control.addWidget(QLabel("Intensity:"))
        intensity_control.addWidget(self.intensity_slider)
        self.intensity_value = QLabel(f"{self.engine.params['intensity']:.2f}")
        intensity_control.addWidget(self.intensity_value)
        intensity_control.addStretch()

        intensity_layout.addLayout(intensity_control)
        intensity_layout.addWidget(QLabel(
            "How strongly the background is removed.\n"
            "0 % = no change, 100 % = full effect."))
        layout.addWidget(intensity_group)

        # ------------------------------------------------------------------
        # Outlier Rejection
        # ------------------------------------------------------------------
        sigma_group = QGroupBox("Outlier Rejection")
        sigma_layout = QVBoxLayout(sigma_group)

        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(10, 50)   # 1.0 – 5.0
        self.sigma_slider.setValue(int(DEFAULT_PARAMS['sigma_threshold'] * 10))

        sigma_control = QHBoxLayout()
        sigma_control.addWidget(QLabel("Sigma Threshold:"))
        sigma_control.addWidget(self.sigma_slider)
        self.sigma_value = QLabel(f"{DEFAULT_PARAMS['sigma_threshold']:.1f}")
        sigma_control.addWidget(self.sigma_value)
        sigma_control.addStretch()

        sigma_layout.addLayout(sigma_control)
        sigma_layout.addWidget(QLabel(
            "Threshold for sigma clipping (1.0‑5.0).\n"
            "Higher values preserve more pixels as background."))
        layout.addWidget(sigma_group)

        # ------------------------------------------------------------------
        # Background Contrast
        # ------------------------------------------------------------------
        median_group = QGroupBox("Background Contrast")
        median_layout = QVBoxLayout(median_group)

        self.median_slider = QSlider(Qt.Orientation.Horizontal)
        self.median_slider.setRange(3, 15)
        self.median_slider.setValue(DEFAULT_PARAMS['median_kernel_size'])
        self.median_slider.setTickInterval(2)

        median_control = QHBoxLayout()
        median_control.addWidget(QLabel("Median Kernel:"))
        median_control.addWidget(self.median_slider)
        self.median_value = QLabel(str(DEFAULT_PARAMS['median_kernel_size']))
        median_control.addWidget(self.median_value)
        median_control.addStretch()

        median_layout.addLayout(median_control)
        median_layout.addWidget(QLabel(
            "Kernel size for median filtering (odd numbers only).\n"
            "Larger kernels smooth more but may lose detail."))
        layout.addWidget(median_group)

        # ------------------------------------------------------------------
        # Wavelet Decomposition 
        # ------------------------------------------------------------------
        wavelet_group = QGroupBox("Wavelet Decomposition")
        wavelet_layout = QVBoxLayout(wavelet_group)

        self.wavelet_slider = QSlider(Qt.Orientation.Horizontal)
        self.wavelet_slider.setRange(3, 10)
        self.wavelet_slider.setValue(DEFAULT_PARAMS['wavelet_levels'])
        self.wavelet_slider.setTickInterval(1)

        wavelet_control = QHBoxLayout()
        wavelet_control.addWidget(QLabel("Levels:"))
        wavelet_control.addWidget(self.wavelet_slider)
        self.wavelet_value = QLabel(str(DEFAULT_PARAMS['wavelet_levels']))
        wavelet_control.addWidget(self.wavelet_value)
        wavelet_control.addStretch()

        wavelet_layout.addLayout(wavelet_control)
        wavelet_layout.addWidget(QLabel(
            "Number of wavelet decomposition levels (3‑10).\n"
            "Higher values capture more detail but increase processing time."))
        layout.addWidget(wavelet_group)

        # ------------------------------------------------------------------
        # Noise Reduction
        # ------------------------------------------------------------------
        denoise_group = QGroupBox("Noise Reduction")
        denoise_layout = QVBoxLayout(denoise_group)

        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(0, 100)   # 0.0 – 1.0
        self.denoise_slider.setValue(int(DEFAULT_PARAMS['denoise_threshold'] * 100))

        denoise_control = QHBoxLayout()
        denoise_control.addWidget(QLabel("Denoise Strength:"))
        denoise_control.addWidget(self.denoise_slider)
        self.denoise_value = QLabel(f"{DEFAULT_PARAMS['denoise_threshold']:.2f}")
        denoise_control.addWidget(self.denoise_value)
        denoise_control.addStretch()

        denoise_layout.addLayout(denoise_control)
        denoise_layout.addWidget(QLabel(
            "Strength of wavelet denoising (0.0‑1.0).\n"
            "Higher values remove more noise but may soften details."))
        layout.addWidget(denoise_group)

        # ------------------------------------------------------------------
        # Dark Mask
        # ------------------------------------------------------------------
        mask_group = QGroupBox("Dark Mask")
        mask_layout = QVBoxLayout(mask_group)

        self.mask_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_slider.setRange(0, 100)
        self.mask_slider.setValue(int(DEFAULT_PARAMS['dark_mask_opacity'] * 100))
        self.mask_slider.setEnabled(True)

        mask_control = QHBoxLayout()
        mask_control.addWidget(QLabel("Opacity:"))
        mask_control.addWidget(self.mask_slider)
        self.mask_value = QLabel(f"{DEFAULT_PARAMS['dark_mask_opacity']:.2f}")
        mask_control.addWidget(self.mask_value)
        mask_control.addStretch()

        mask_layout.addLayout(mask_control)
        mask_layout.addWidget(QLabel(
            "Optional darkening mask to further reduce background brightness.\n"
            "Applies radial gradient from center to edges."))
        layout.addWidget(mask_group)

        layout.addStretch()
        return widget


    # ------------------------------------------------------------------
    # Help dialog
    # ------------------------------------------------------------------
    def _show_help(self):
        """Display the help tab in a modal dialog."""
        help_widget = self._create_help_tab()
        dlg = QDialog(self)
        dlg.setWindowTitle("Help")
        layout = QVBoxLayout(dlg)
        layout.addWidget(help_widget)
        dlg.resize(600, 550)
        dlg.exec()

    def _create_help_tab(self) -> QWidget:
        """Create the help/information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        help_text = """
        <h3 style="color: #88aaff;">About This Tool</h3>
        <p>This tool implements astrophotography image optimization techniques.
        It uses wavelet decomposition and advanced filtering to harmonize 
        the sky background while preserving astronomical details.</p>
        <h4 style="color: #88aaff;">How It Works</h4>
        <ul>
        <li><b>Background Intensity:</b> Identifies and adjusts the sky background darkness</li>
        <li><b>Outlier Rejection:</b> Adjusts background preservation</li>
        <li><b>Background Contrast:</b> Increases contrast between light and dark tones</li>
        <li><b>Wavelet Decomposition:</b> Breaks the image into frequency components for sharpening</li>
        <li><b>Noise Reduction:</b> Reduces noise in the wavelet domain</li>
        <li><b>RDark Mask:</b> Applie a radial dark mask to soften the image</li>
        </ul>
        <h4 style="color: #88aaff;">Usage Tips</h4>
        <ul>
        <li>Start with default parameters for most images</li>
        <li>Use live preview to adjust parameters interactively</li>
        <li>For noisy images, increase denoising strength</li>
        <li>For complex backgrounds, increase wavelet levels</li>
        <li>The dark mask is optional and should be used sparingly</li>
        </ul>
        <h4 style="color: #88aaff;">Performance</h4>
        <p>Processing time depends on image size and parameters. The live preview
        uses a downscaled version for real‑time feedback. Full resolution
        processing is applied when you click "Process Full Image".</p>
        <h4 style="color: #88aaff;">Credits</h4>
        <p>Developed for SIRIL.<br>
        (c) G. Trainar (2026)</p>
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #cccccc; font-size: 12pt;")
        layout.addWidget(help_label)
        return widget

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------
    def _setup_connections(self):
        self.wavelet_slider.valueChanged.connect(
            lambda val: self._on_param_change('wavelet_levels', val))
        self.median_slider.valueChanged.connect(
            lambda val: self._on_param_change('median_kernel_size', val))
        self.sigma_slider.valueChanged.connect(
            lambda val: self._on_param_change('sigma_threshold', val / 10.0))
        self.denoise_slider.valueChanged.connect(
            lambda val: self._on_param_change('denoise_threshold', val / 100.0))
        self.intensity_slider.valueChanged.connect(
            lambda val: self._on_param_change('intensity', val / 100.0))

        self.mask_slider.valueChanged.connect(self._on_dark_mask_value_changed)

        self._preview_timer.timeout.connect(self._start_preview)

        self.preview_worker.preview_ready.connect(
            self._update_preview,
            Qt.ConnectionType.QueuedConnection)
        self.preview_worker.processing_time.connect(
            self._handle_processing_time)

        # Zoom / fit buttons
        self.zoom_in_btn.clicked.connect(self.preview_label.zoomIn)
        self.zoom_out_btn.clicked.connect(self.preview_label.zoomOut)
        self.fit_btn.clicked.connect(self.preview_label.fitToWindow)
        self.one_to_one_btn.clicked.connect(self.preview_label.resetZoom)

    def _on_param_change(self, param_name: str, value):
        """Update engine parameter and restart preview debounce timer."""
        self.engine.set_parameters(**{param_name: value})

        if param_name == 'wavelet_levels':
            self.wavelet_value.setText(str(value))
        elif param_name == 'median_kernel_size':
            self.median_value.setText(str(value))
        elif param_name == 'sigma_threshold':
            self.sigma_value.setText(f"{value:.1f}")
        elif param_name == 'denoise_threshold':
            self.denoise_value.setText(f"{value:.2f}")
        elif param_name == 'dark_mask_opacity':
            self.mask_value.setText(f"{value:.2f}")
        elif param_name == 'intensity':
            self.intensity_value.setText(f"{value:.2f}")
        elif param_name == 'edge_smooth_mod':
            self.edge_value.setText(f"{value:.2f}")

        # Restart debounce timer
        self._preview_timer.start(200)

    # Dark‑mask debounce helpers
    def _on_dark_mask_value_changed(self, val):
        self._dark_mask_timer.start(200)  

    def _apply_dark_mask_preview(self):
        val = self.mask_slider.value()
        self.engine.set_parameters(dark_mask_opacity=val / 100.0)

        self.mask_value.setText(f"{val / 100.0:.2f}")

        if hasattr(self, 'preview_worker') and self.preview_worker.isRunning():
            self.preview_worker.set_image(self.current_image)

    def _handle_processing_time(self, t):
        """Store the last processing time for progress bar logic."""
        self.last_processing_time = t

    # Resize‑timer helpers
    def resizeEvent(self, event) -> None:
        """Simply rescale the preview on resize."""
        super().resizeEvent(event)
        self._rescale_preview()
        if not getattr(self.preview_label, 'manual_zoom', False):
            self.preview_label.fitToWindow()

    def _restart_preview_after_resize(self) -> None:
        """Not used – preview always runs."""
        pass

    def _rescale_preview(self):
        """Re‑scale the preview image to fit the current size of the
        preview widget.  Called after a resize event and after each
        preview update."""
        if self.original_image is None:
            return

        try:
            if self.show_original or self.engine.params.get('intensity', 0.0) == 0:
                img_to_show = np.flipud(self.original_image)
            else:
                if self.engine.last_processed is not None:
                    orig_h, orig_w = self.original_image.shape[:2]
                    processed_up = cv2.resize(
                        np.flipud(self.engine.last_processed),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_AREA)

                    intensity = self.engine.params.get('intensity', 0.0)
                    orig_resized = cv2.resize(
                        np.flipud(self.original_image),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_AREA)
                    img_to_show = (1 - intensity) * orig_resized + \
                                intensity * processed_up
                else:
                    img_to_show = np.flipud(self.original_image)

            self._display_image(img_to_show, fit=False)

            self.original_indicator.setVisible(self.show_original)

        except Exception as e:
            print(f"Rescale preview error: {e}")

    def _display_image(self, img: np.ndarray, fit: bool = True):
        """Render the given NumPy image in the preview label."""
        qimg = self._numpy_to_qimage(img)
        self.preview_label.setBaseImage(qimg)
        if fit:
            self.preview_label.fitToWindow()
        else:
            self.preview_label.updatePixmap()

        if not self._first_preview_done:
            self.preview_label.fitToWindow()
            self._first_preview_done = True

    # Image handling
    def _load_image(self):
        try:
            try:
                current_fname = self.siril.get_image_filename()
                def is_tiff(fn):
                    return fn.lower().endswith(('.tif', '.tiff'))

                if not is_tiff(current_fname):
                    try:
                        dir_name, base = os.path.split(current_fname)
                        new_base = os.path.splitext(base)[0]
                        new_path = os.path.join(dir_name, f"{new_base}")

                        self.siril.cmd(f'savetif32 {shlex.quote(new_path)}')
                        self.siril.cmd(f'load {shlex.quote(new_path)}')
                        current_fname = new_path
                    except Exception as e:
                        raise RuntimeError(f"Could not convert to 32‑bit TIFF: {e}")

                img = self.siril.get_image()
                if img is None:
                    raise RuntimeError("No image data received from SIRIL")

                image_data = img.data
                if image_data.dtype == np.uint16:
                    image_data = image_data.astype(np.float32) / 65535.0
                elif image_data.dtype == np.uint8:
                    image_data = image_data.astype(np.float32) / 255.0
                else:
                    image_data = image_data.astype(np.float32)
                if image_data.ndim == 3 and image_data.shape[0] == 3:
                    image_data = np.transpose(image_data, (1, 2, 0))
                elif image_data.ndim == 2:
                    image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)

                self.status_label.setText("Image loaded from SIRIL")
                self.status_label.setStyleSheet("color: #88ff88;")

            except Exception as siril_error:
                file_dialog = QFileDialog()
                file_dialog.setNameFilter("Images (*.tif *.tiff *.fit *.fits *.png *.jpg)")
                file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

                if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                    file_path = file_dialog.selectedFiles()[0]
                    self.siril.cmd(f'load {shlex.quote(file_path)}')
                    img = self.siril.get_image()
                    if img is None:
                        raise RuntimeError(f"Failed to load image: {file_path}")

                    image_data = img.data
                    if image_data.ndim == 3 and image_data.shape[0] == 3:
                        image_data = np.transpose(image_data, (1, 2, 0))
                    elif image_data.ndim == 2:
                        image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)
                        

                    self.status_label.setText(f"Image loaded: {os.path.basename(file_path)}")
                    self.status_label.setStyleSheet("color: #88ff88;")

                else:
                    return

            self.original_image = image_data.copy()
            self.current_image = image_data.copy()
            self._display_image(np.flipud(self.original_image), fit=True)

            if 'current_fname' in locals():
                self.original_file_path = current_fname
            else:
                self.original_file_path = file_path

            self._first_preview_done = False
            self.show_original = False

            self._start_preview()
            self.btn_process.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error loading image: {e}")
            self.status_label.setStyleSheet("color: #ff8888;")
            print(f"Load image error: {e}")
            traceback.print_exc()

    def _start_preview(self):
        """Start or update the preview worker with the current image."""
        if self.current_image is None:
            return

        if hasattr(self, "preview_worker") and self.preview_worker.isRunning():
            self.preview_worker.set_image(self.current_image)
            return

        # Re‑create the worker to pass the preview label
        self.preview_worker = PreviewWorker(self.engine, self.preview_label)

        self.preview_worker.preview_ready.connect(
            self._update_preview,
            Qt.ConnectionType.QueuedConnection)
        self.preview_worker.processing_time.connect(
            self._handle_processing_time)

        self.preview_worker.set_image(self.current_image)
        self.preview_worker.start()

    def _update_preview(self, preview_data):
        """Update the preview widgets with new data."""
        if self._is_resizing:
            return

        try:
            self.engine.last_processed = preview_data

            preview_flipped = np.flipud(preview_data)

            if self.show_original or self.engine.params.get('intensity', 0.0) == 0:
                img_to_show = np.flipud(self.original_image)
            else:
                intensity = self.engine.params.get('intensity', 0.0)
                if intensity > 0:
                    orig_h, orig_w = self.original_image.shape[:2]
                    processed_up = cv2.resize(
                        preview_flipped,
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_AREA)

                    orig_resized = cv2.resize(
                        np.flipud(self.original_image),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_AREA)
                    img_to_show = (1 - intensity) * orig_resized + \
                                intensity * processed_up
                else:
                    img_to_show = preview_flipped

            self._display_image(img_to_show, fit=False)

            self.original_indicator.setVisible(self.show_original)

            t = getattr(self, 'last_processing_time', 0.0)
            if t > 1.0:
                # Show progress bar at bottom
                self.progress_bar.setVisible(True)
                self.overlay_widget.updateProgressBarPosition()
                QTimer.singleShot(int(t * 1000), lambda: self.progress_bar.setVisible(False))

        except Exception as e:
            print(f"Preview update error: {e}")

    def _numpy_to_qimage(self, data: np.ndarray) -> QImage:
        """Convert a NumPy array in [0,1] to a QImage (RGB)."""
        if data is None:
            return QImage()

        data = np.clip(data, 0.0, 1.0)
        data_8bit = (data * 255).astype(np.uint8)

        h, w = data_8bit.shape[:2]
        return QImage(
            data_8bit.tobytes(),
            w,
            h,
            data_8bit.strides[0],
            QImage.Format.Format_RGB888)

    def _process_full_image(self):
        try:
            if self.current_image is None:
                raise RuntimeError("No image loaded")

            self.status_label.setText("Processing full image…")
            self.status_label.setStyleSheet("color: #ffcc00;")
            self.app.processEvents()

            result = self.engine.process_image(self.current_image)
            self.current_image = result
            self.show_original = False
            self.engine.last_processed = result  
            self._display_image(np.flipud(result), fit=False)

            self.status_label.setText("Processing complete!")
            self.status_label.setStyleSheet("color: #88ff88;")
            self.btn_save.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Processing failed: {e}")
            self.status_label.setStyleSheet("color: #ff8888;")
            print(f"Processing error: {e}")
            traceback.print_exc()

    def _save_result(self):
        try:
            if self.current_image is None:
                raise RuntimeError("No processed image to save")

            try:
                if self.current_image.ndim == 3:
                    siril_data = np.transpose(self.current_image, (2, 0, 1))
                else:
                    siril_data = self.current_image

                siril_data = siril_data.astype(np.float32)

                with self.siril.image_lock():
                    self.siril.set_image_pixeldata(siril_data)

                if self.original_file_path:
                    base, _ = os.path.splitext(self.original_file_path)
                    output_name = f"{base}_ap_processed"
                else:
                    output_name = "ap_processed"
                self.siril.cmd("save", f'"{output_name}"')

                self.status_label.setText(f"Saved to SIRIL: {output_name}")
                self.status_label.setStyleSheet("color: #88ff88;")

            except Exception as siril_error:
                file_dialog = QFileDialog()
                file_dialog.setNameFilter("TIFF Images (*.tif *.tiff)")
                file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
                file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                file_dialog.setDefaultSuffix("tif")

                if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                    file_path = file_dialog.selectedFiles()[0]
                    result_16bit = (self.current_image * 65535).astype(np.uint16)
                    cv2.imwrite(file_path, result_16bit)

                    self.status_label.setText(f"Saved: {file_path}")
                    self.status_label.setStyleSheet("color: #88ff88;")

        except Exception as e:
            self.status_label.setText(f"Save failed: {e}")
            self.status_label.setStyleSheet("color: #ff8888;")
            print(f"Save error: {e}")

    def closeEvent(self, event):
        if self.preview_worker.isRunning():
            self.preview_worker.stop()
        event.accept()

    # Keyboard handling for toggle preview
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.show_original = not self.show_original
            self._rescale_preview()
        else:
            super().keyPressEvent(event)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    try:
        if not ensure_dependencies():
            print("Failed to install required dependencies.")
            return

        import cv2
        from skimage import color, filters
        from scipy import signal

        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        siril = s.SirilInterface()
        try:
            siril.connect()
        except Exception as e:
            QMessageBox.critical(None, "SIRIL Connection Error",
                                 f"Could not connect to SIRIL.\n{e}")
            return

        gui = AstroPhotoProcessorGUI(siril, app)
        gui.show()
        sys.exit(app.exec())

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
