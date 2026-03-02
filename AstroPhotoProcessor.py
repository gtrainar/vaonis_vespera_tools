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

s.ensure_installed("PyQt6", "opencv-python", "scikit-image", "scipy", "astropy")

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
    'sigma_threshold': 1.0,
    'denoise_threshold': 0.0,
    'dark_mask_opacity': 0.0,
    'preview_scale': 0.75, 
    'intensity': 0.0, 
    # Image levels / stretch
    'black_point': 0.0,       # 0.0 – 0.5  (fraction of full range)
    'white_point': 1.0,       # 0.5 – 1.0
    'stretch_intensity': 1.0, # 0.5 – 3.0  (gamma-like midtone stretch)
    # Color
    'vibrance': 0.0,    # -1.0 – 1.0
    'saturation': 0.0,  # -1.0 – 1.0
    'contrast': 0.0,    # -1.0 – 1.0
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
        self.reference_size: int = 0  # shorter dimension of the full-res image

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

    def set_reference_image(self, image_data: np.ndarray):
        """Record the full-resolution image dimensions so that scale-dependent
        parameters (median kernel, Gaussian sigma) produce equivalent spatial
        behaviour on both the downscaled preview and the full-res render."""
        h, w = image_data.shape[:2]
        self.reference_size = min(h, w)

    def process_image(self, image_data: np.ndarray,
                  zoom_factor: float = 1.0) -> np.ndarray:
        """Main processing pipeline (zoom_factor only used for smoothing)."""
        start_time = time.time()
        try:
            if image_data.dtype != np.float32:
                image_data = image_data.astype(np.float32)

            # 1. Background Intensity
            intensity = self.params.get('intensity', 0.0)

            # Convert to Lab color space
            lab_image = self._rgb_to_lab(image_data)
            luminance = lab_image[:, :, 0]
            original_shape = luminance.shape

            # 2. Star Protection
            background = self._estimate_background(luminance)

            # 3. Smoothing 
            processed_luminance = luminance - background

            # 4. Wavelet Decomposition 
            wavelet_scales = self._wavelet_decompose(processed_luminance)
            processed_scales = self._process_background(wavelet_scales)

            # 5. Noise Reduction
            processed_luminance = self._wavelet_reconstruct(processed_scales, original_shape)

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

            # 6. Levels & stretch (black point / white point / midtone)
            result_rgb = self._apply_levels(result_rgb)

            # 7. Color (vibrance / saturation / contrast)
            result_rgb = self._apply_color(result_rgb)

            # 8. Dark Mask
            if self.params['dark_mask_opacity'] > 0:
                result_rgb = self._apply_dark_mask(result_rgb)  # apply to the RGB output

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
        h, w = approximation.shape[:2]
        current_size = min(h, w)

        # Scale the kernel and Gaussian sigma proportionally to the image size
        # so that background estimation is spatially equivalent on both the
        # downscaled preview and the full-resolution render.
        if self.reference_size > 0 and current_size > 0:
            scale = current_size / self.reference_size
        else:
            scale = 1.0

        kernel = self.params['median_kernel_size']
        kernel = max(3, int(round(kernel * scale)))
        if kernel % 2 == 0:
            kernel += 1

        sigma = max(0.5, 1.0 * scale)

        try:
            from skimage.morphology import footprint_rectangle
            footprint = footprint_rectangle((kernel, kernel))
        except ImportError:
            from skimage.morphology import rectangle
            footprint = rectangle(kernel, kernel)

        median_filtered = filters.median(approximation, footprint=footprint)
        blurred = filters.gaussian(median_filtered, sigma=sigma)

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
    
        # Normalized distance: 0 at center, 1 at corners
        norm_dist = distance / max_dist
        # Darken edges: mask is 1.0 at center, falls toward (1-opacity) at edges
        mask = 1.0 - opacity * norm_dist  # ← was wrongly using (1 - norm_dist)

        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
        return image * mask

    def _apply_levels(self, image: np.ndarray) -> np.ndarray:
        """Apply black-point, white-point and midtone stretch."""
        bp = float(self.params.get('black_point', 0.0))
        wp = float(self.params.get('white_point', 1.0))
        si = float(self.params.get('stretch_intensity', 1.0))
        # Guard against degenerate range
        if wp <= bp:
            wp = bp + 1e-6
        # Remap [bp, wp] → [0, 1]
        out = (image - bp) / (wp - bp)
        # Midtone power stretch (si > 1 brightens, < 1 darkens)
        out = np.power(np.clip(out, 0.0, 1.0), 1.0 / si)
        return out

    def _apply_color(self, image: np.ndarray) -> np.ndarray:
        """Apply vibrance, saturation and contrast adjustments in Lab space."""
        vibrance   = float(self.params.get('vibrance',   0.0))
        saturation = float(self.params.get('saturation', 0.0))
        contrast   = float(self.params.get('contrast',   0.0))
        if vibrance == 0.0 and saturation == 0.0 and contrast == 0.0:
            return image

        lab = self._rgb_to_lab(np.clip(image, 0.0, 1.0))
        L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]

        # Saturation: scale a/b channels uniformly
        sat_factor = 1.0 + saturation
        a = a * sat_factor
        b = b * sat_factor

        # Vibrance: boost low-saturation pixels more gently
        if vibrance != 0.0:
            chroma = np.sqrt(a**2 + b**2)
            # pixels with low chroma get more boost
            vib_factor = 1.0 + vibrance * (1.0 - chroma / (chroma.max() + 1e-6))
            a = a * vib_factor
            b = b * vib_factor

        # Contrast: S-curve on the L channel around mid-grey (L=50)
        if contrast != 0.0:
            L = 50.0 + (L - 50.0) * (1.0 + contrast)
            L = np.clip(L, 0.0, 100.0)

        lab_out = np.stack([L, a, b], axis=2)
        return np.clip(self._lab_to_rgb(lab_out), 0.0, 1.0)

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

    def __init__(self, engine, preview_label):
        super().__init__()
        self.engine = engine
        self.preview_label = preview_label
        self.full_image = None
        self._is_running = False
        self._lock = threading.Lock()
        self._needs_update = threading.Event()  # ← ADD THIS

    def set_image(self, image_data):
        with self._lock:
            self.full_image = image_data
        self._needs_update.set()  # ← signal update needed

    def request_update(self):
        """Called when params change but image hasn't."""
        self._needs_update.set()
    
    def stop(self):
        self._is_running = False
        self.wait()
        with self._lock:
            self.full_image = None

    def run(self):
        self._is_running = True
        while self._is_running:
            triggered = self._needs_update.wait(timeout=0.5)
            if not triggered:
                continue
            self._needs_update.clear()
            with self._lock:
                if self.full_image is not None:
                    try:
                        preview_image = self._create_preview()
                        self.preview_ready.emit(preview_image)
                        self.processing_time.emit(self.engine.processing_time)
                    except Exception as e:
                        print(f"Preview error: {e}")
                        break

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
        # Temporarily cap wavelet levels for preview
        original_levels = self.engine.params['wavelet_levels']
        h, w = small_img.shape[:2]
        max_useful = min(original_levels, (min(h, w).bit_length() - 2))
        self.engine.params['wavelet_levels'] = max_useful
        result = self.engine.process_image(small_img, zoom_factor=zoom)
        self.engine.params['wavelet_levels'] = original_levels  # restore
        return result


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

        # Always render onto a fixed-size canvas equal to the viewport.
        # Using a single constant alignment (AlignLeft|AlignTop) prevents the
        # top-left jump caused by switching between AlignCenter and
        # AlignLeft|AlignTop depending on image/zoom size at each update.
        canvas = QPixmap(view_w, view_h)
        canvas.fill(Qt.GlobalColor.black)
        painter = QPainter(canvas)
        if w <= view_w and h <= view_h:
            # Image fits entirely – draw centred, reset pan offsets
            draw_x = (view_w - w) // 2
            draw_y = (view_h - h) // 2
            self.offset_x = 0
            self.offset_y = 0
            painter.drawPixmap(draw_x, draw_y, pixmap)
        else:
            # Image larger than viewport – use the offset/pan system
            max_x = max(0, w - view_w)
            max_y = max(0, h - view_h)
            self.offset_x = max(0, min(self.offset_x, max_x))
            self.offset_y = max(0, min(self.offset_y, max_y))
            src_rect = QRect(self.offset_x, self.offset_y, view_w, view_h)
            painter.drawPixmap(0, 0, pixmap.copy(src_rect))
        painter.end()

        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setPixmap(canvas)


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

        # Auto-load the image currently open in SIRIL (deferred so the window
        # is fully painted before the loading work starts).
        QTimer.singleShot(0, self._load_from_siril)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _get_dark_stylesheet(self) -> str:
        return """
        QDialog { background-color: #2b2b2b; color: #e0e0e0; }
        QGroupBox { border: 1px solid #444444; margin-top: 12px;
                    font-weight: bold; border-radius: 4px; padding-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px;
                           padding: 0 5px; color: #88aaff; }
        QLabel { color: #cccccc; font-size: 12pt; }
        QLabel#title { color: #88aaff; font-size: 16pt; font-weight: bold; }
        QSlider::groove:horizontal { border: 1px solid #555555; height: 8px;
                                     background: #3c3c3c; border-radius: 4px; }
        QSlider::handle:horizontal { background: #88aaff; border: 1px solid #555555;
                                     width: 16px; height: 16px; margin: -4px 0;
                                     border-radius: 8px; }
        QPushButton { background-color: #444444; color: #dddddd;
                      border: 1px solid #666666; border-radius: 4px;
                      padding: 8px 20px; font-weight: bold; min-width: 100px; }
        QPushButton:hover { background-color: #555555; }
        QPushButton:pressed { background-color: #333333; }
        QPushButton:disabled { background-color: #333333; color: #666666;
                               border: 1px solid #444444; }
        QPushButton#process { background-color: #285299; }
        QPushButton#process:hover { background-color: #3363bb; }
        QPushButton#process:disabled { background-color: #1a3360; color: #556688; }
        QPushButton#close_btn { background-color: #553333; }
        QPushButton#close_btn:hover { background-color: #774444; }
        QStatusBar { background-color: #222222; color: #ffcc00;
                     font-size: 10pt; border-top: 1px solid #444444; }
        """

    def _build_ui(self) -> None:
        """Create the entire dialog layout – called from __init__."""
        self.setWindowTitle(f"{SCRIPT_NAME} v{VERSION}")

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

        self.left_container = QGroupBox()
        left_container_layout = QVBoxLayout(self.left_container)
        left_container_layout.setContentsMargins(10, 10, 10, 10)
        left_container_layout.addWidget(self._create_processing_tab())
        left_layout.addWidget(self.left_container)

        # ------------------------------------------------------------------
        # Right side – preview
        # ------------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        preview_group = QGroupBox()
        preview_layout = QVBoxLayout(preview_group)

        # Zoomable preview label
        self.preview_label = ZoomableLabel()
        self.preview_label.setMinimumSize(512, 512)
        self.preview_label.setStyleSheet(
            "background-color: #111; border: 1px solid #444;"
        )
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_label.setToolTip(
            "Scroll wheel or Zoom +/– to zoom · Drag to pan · Space = toggle original"
        )

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #333; color: white; border: 1px solid #555; "
            "text-align: center; } QProgressBar::chunk { background-color: #88aaff; }"
        )
        self.progress_bar.setFixedHeight(10)

        self.overlay_widget = OverlayGraphicsView(self.scroll_area, self.progress_bar)
        preview_layout.addWidget(self.overlay_widget)

        # ORIGINAL indicator overlay
        self.original_indicator = QLabel("● ORIGINAL", self.preview_label)
        self.original_indicator.setStyleSheet("""
            background-color: rgba(0, 0, 0, 160);
            color: #ffcc00;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
        """)
        self.original_indicator.move(10, 10)
        self.original_indicator.setVisible(False)
        self.original_indicator.adjustSize()

        # Zoom / fit buttons row — with keyboard shortcut hints in tooltips
        zoom_btns = QHBoxLayout()
        self.zoom_out_btn   = QPushButton("Zoom –")
        self.zoom_in_btn    = QPushButton("Zoom +")
        self.fit_btn        = QPushButton("Fit")
        self.one_to_one_btn = QPushButton("1:1")

        self.zoom_out_btn.setToolTip("Zoom out (scroll wheel down)")
        self.zoom_in_btn.setToolTip("Zoom in (scroll wheel up)")
        self.fit_btn.setToolTip("Fit entire image in the viewport")
        self.one_to_one_btn.setToolTip("Show actual pixels — 1 image pixel = 1 screen pixel")

        # Space bar hint label, right-aligned next to zoom buttons
        space_hint = QLabel("Space: toggle original")
        space_hint.setStyleSheet("color: #ff8800; font-size: 9pt; font-style: italic;")
        space_hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        zoom_btns.addWidget(self.zoom_out_btn)
        zoom_btns.addWidget(self.zoom_in_btn)
        zoom_btns.addWidget(self.fit_btn)
        zoom_btns.addWidget(self.one_to_one_btn)
        zoom_btns.addStretch()
        zoom_btns.addWidget(space_hint)

        preview_layout.addLayout(zoom_btns)
        right_layout.addWidget(preview_group)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # ------------------------------------------------------------------
        # Main dialog layout
        # ------------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter fills the window
        main_layout.addWidget(splitter)

        # ------------------------------------------------------------------
        # Bottom action bar  (Load | Reset | Render Full Res | Save — Help | Close)
        # ------------------------------------------------------------------
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(52)
        bottom_bar.setStyleSheet("background-color: #222222; border-top: 1px solid #444444;")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 8, 12, 8)
        bottom_layout.setSpacing(8)

        self.btn_load = QPushButton("⏏  Load Image")
        self.btn_load.setToolTip("Load the current SIRIL image or browse for a file")

        self.btn_reset = QPushButton("↺  Reset")
        self.btn_reset.setToolTip("Load an image first")
        self.btn_reset.setEnabled(False)

        self.btn_process = QPushButton("⚙  Render Full Resolution")
        self.btn_process.setObjectName("process")
        self.btn_process.setToolTip("Load an image first")
        self.btn_process.setEnabled(False)

        self.btn_save = QPushButton("💾  Save Result")
        self.btn_save.setToolTip("Render the full-resolution image first")
        self.btn_save.setEnabled(False)

        # Status label lives in the bottom bar, left-aligned
        self.status_label = QLabel("Ready — load an image to begin")
        self.status_label.setObjectName("status")
        self.status_label.setStyleSheet("color: #ffcc00; font-size: 10pt; background: transparent;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.help_btn = QPushButton("?  Help")
        self.help_btn.setToolTip("Show usage instructions")
        self.help_btn.setMinimumWidth(80)

        self.btn_close = QPushButton("✕  Close")
        self.btn_close.setObjectName("close_btn")
        self.btn_close.setToolTip("Close this dialog")
        self.btn_close.setMinimumWidth(80)

        bottom_layout.addWidget(self.btn_load)
        bottom_layout.addWidget(self.btn_reset)
        bottom_layout.addWidget(self.btn_process)
        bottom_layout.addWidget(self.btn_save)
        bottom_layout.addSpacing(16)
        bottom_layout.addWidget(self.status_label, 1)  # stretch to fill middle
        bottom_layout.addSpacing(16)
        bottom_layout.addWidget(self.help_btn)
        bottom_layout.addWidget(self.btn_close)

        main_layout.addWidget(bottom_bar)


    # ------------------------------------------------------------------
    # Create processing tab (sliders, checkboxes)
    # ------------------------------------------------------------------
    def _create_processing_tab(self) -> QWidget:
        widget = QWidget()
        # Two-column layout: left col | right col
        columns = QHBoxLayout(widget)
        columns.setSpacing(10)
        columns.setContentsMargins(0, 0, 0, 0)

        left_col_widget = QWidget()
        left_col_widget.setMinimumWidth(180)
        left_col_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        left_col = QVBoxLayout(left_col_widget)
        left_col.setSpacing(10)
        left_col.setContentsMargins(0, 0, 0, 0)

        right_col_widget = QWidget()
        right_col_widget.setMinimumWidth(180)
        right_col_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_col = QVBoxLayout(right_col_widget)
        right_col.setSpacing(10)
        right_col.setContentsMargins(0, 0, 0, 0)

        # Helper: build a label-above / full-width-slider row inside a layout
        def add_slider_row(parent_layout, label_text, slider, value_label):
            header = QHBoxLayout()
            header.addWidget(QLabel(label_text))
            header.addStretch()
            header.addWidget(value_label)
            parent_layout.addLayout(header)
            parent_layout.addWidget(slider)

        # ==============================================================
        # LEFT COLUMN
        # ==============================================================

        # ------------------------------------------------------------------
        # Image Levels
        # ------------------------------------------------------------------
        levels_group = QGroupBox("Levels")
        levels_layout = QVBoxLayout(levels_group)
        levels_layout.setSpacing(4)

        self.black_slider = QSlider(Qt.Orientation.Horizontal)
        self.black_slider.setRange(0, 50)
        self.black_slider.setValue(int(DEFAULT_PARAMS['black_point'] * 100))
        self.black_slider.setToolTip(
            'Darkness – sets the black point.\n'
            'Adjusts how dark the sky background appears.\n'
            'Range 0.00 – 0.50')
        self.black_value = QLabel(f"{DEFAULT_PARAMS['black_point']:.2f}")
        add_slider_row(levels_layout, 'Darkness', self.black_slider, self.black_value)

        self.white_slider = QSlider(Qt.Orientation.Horizontal)
        self.white_slider.setRange(50, 100)
        self.white_slider.setValue(int(DEFAULT_PARAMS['white_point'] * 100))
        self.white_slider.setToolTip(
            'Highlights – sets the white point.\n'
            'Controls the intensity of the brightest areas.\n'
            'Range 0.50 – 1.00')
        self.white_value = QLabel(f"{DEFAULT_PARAMS['white_point']:.2f}")
        add_slider_row(levels_layout, 'Highlights', self.white_slider, self.white_value)

        self.stretch_slider = QSlider(Qt.Orientation.Horizontal)
        self.stretch_slider.setRange(50, 300)
        self.stretch_slider.setValue(int(DEFAULT_PARAMS['stretch_intensity'] * 100))
        self.stretch_slider.setToolTip(
            'Stretch Intensity – global midtone stretch.\n'
            'Sets overall contrast between sky background,\n'
            'deep-sky objects, and stars.\n'
            'Range 0.50 – 3.00')
        self.stretch_value = QLabel(f"{DEFAULT_PARAMS['stretch_intensity']:.2f}")
        add_slider_row(levels_layout, 'Intensity', self.stretch_slider, self.stretch_value)

        left_col.addWidget(levels_group)

        # ------------------------------------------------------------------
        # Color
        # ------------------------------------------------------------------
        color_group = QGroupBox("Color")
        color_layout = QVBoxLayout(color_group)
        color_layout.setSpacing(4)

        self.vibrance_slider = QSlider(Qt.Orientation.Horizontal)
        self.vibrance_slider.setRange(-100, 100)
        self.vibrance_slider.setValue(int(DEFAULT_PARAMS['vibrance'] * 100))
        self.vibrance_slider.setToolTip(
            "Vibrance: gently increases or decreases color intensity,\n"
            "preserving subtle tones.\n"
            "Range -1.00 – 1.00")
        self.vibrance_value = QLabel(f"{DEFAULT_PARAMS['vibrance']:.2f}")
        add_slider_row(color_layout, 'Vibrance', self.vibrance_slider, self.vibrance_value)

        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(int(DEFAULT_PARAMS['saturation'] * 100))
        self.saturation_slider.setToolTip(
            "Saturation: strongly increases or decreases overall color intensity\n"
            "for a more dramatic look.\n"
            "Range -1.00 – 1.00")
        self.saturation_value = QLabel(f"{DEFAULT_PARAMS['saturation']:.2f}")
        add_slider_row(color_layout, 'Saturation', self.saturation_slider, self.saturation_value)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(int(DEFAULT_PARAMS['contrast'] * 100))
        self.contrast_slider.setToolTip(
            "Contrast: enhances the difference between shadows and highlights,\n"
            "adding depth to the image.\n"
            "Range -1.00 – 1.00")
        self.contrast_value = QLabel(f"{DEFAULT_PARAMS['contrast']:.2f}")
        add_slider_row(color_layout, 'Contrast', self.contrast_slider, self.contrast_value)

        left_col.addWidget(color_group)

        # ------------------------------------------------------------------
        # Dark Mask
        # ------------------------------------------------------------------
        mask_group = QGroupBox("Dark Mask")
        mask_layout = QVBoxLayout(mask_group)
        mask_layout.setSpacing(4)

        self.mask_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_slider.setRange(0, 100)
        self.mask_slider.setValue(int(DEFAULT_PARAMS['dark_mask_opacity'] * 100))
        self.mask_slider.setToolTip(
            "Optional darkening mask to further reduce background brightness.\n"
            "Applies radial gradient from center to edges.")
        self.mask_value = QLabel(f"{DEFAULT_PARAMS['dark_mask_opacity']:.2f}")
        add_slider_row(mask_layout, 'Opacity', self.mask_slider, self.mask_value)

        left_col.addWidget(mask_group)
        left_col.addStretch()

        # ==============================================================
        # RIGHT COLUMN
        # ==============================================================

        # ------------------------------------------------------------------
        # Background
        # ------------------------------------------------------------------
        background_group = QGroupBox("Background")
        background_layout = QVBoxLayout(background_group)
        background_layout.setSpacing(4)

        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(int(self.engine.params['intensity'] * 100))
        self.intensity_slider.setTickInterval(10)
        self.intensity_slider.setToolTip(
            "How strongly the background is removed.\n"
            "0 ‰ = no change, 100 ‰ = full effect.")
        self.intensity_value = QLabel(f"{self.engine.params['intensity']:.2f}")
        add_slider_row(background_layout, 'Intensity', self.intensity_slider, self.intensity_value)

        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(10, 50)
        self.sigma_slider.setValue(int(DEFAULT_PARAMS['sigma_threshold'] * 10))
        self.sigma_slider.setToolTip(
            "Star Protection (sigma clipping threshold, 1.0‑5.0).\n"
            "Higher values protect more stars and objects\n"
            "from being flattened by background removal.")
        self.sigma_value = QLabel(f"{DEFAULT_PARAMS['sigma_threshold']:.1f}")
        add_slider_row(background_layout, 'Star Protection', self.sigma_slider, self.sigma_value)

        self.median_slider = QSlider(Qt.Orientation.Horizontal)
        self.median_slider.setRange(3, 15)
        self.median_slider.setValue(DEFAULT_PARAMS['median_kernel_size'])
        self.median_slider.setTickInterval(2)
        self.median_slider.setToolTip(
            "Kernel size for median filtering (odd numbers only).\n"
            "Larger kernels smooth more but may lose detail.")
        self.median_value = QLabel(str(DEFAULT_PARAMS['median_kernel_size']))
        add_slider_row(background_layout, 'Smoothing', self.median_slider, self.median_value)

        right_col.addWidget(background_group)

        # ------------------------------------------------------------------
        # Noise Reduction
        # ------------------------------------------------------------------
        noise_group = QGroupBox("Noise Reduction")
        noise_layout = QVBoxLayout(noise_group)
        noise_layout.setSpacing(4)

        self.wavelet_slider = QSlider(Qt.Orientation.Horizontal)
        self.wavelet_slider.setRange(3, 10)
        self.wavelet_slider.setValue(DEFAULT_PARAMS['wavelet_levels'])
        self.wavelet_slider.setTickInterval(1)
        self.wavelet_slider.setToolTip(
            "Number of wavelet decomposition levels (3‑10).\n"
            "Higher values capture more detail but increase processing time.")
        self.wavelet_value = QLabel(str(DEFAULT_PARAMS['wavelet_levels']))
        add_slider_row(noise_layout, 'Wavelet Decomposition', self.wavelet_slider, self.wavelet_value)

        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setValue(int(DEFAULT_PARAMS['denoise_threshold'] * 100))
        self.denoise_slider.setToolTip(
            "Strength of wavelet denoising (0.0‑1.0).\n"
            "Higher values remove more noise but may soften details.")
        self.denoise_value = QLabel(f"{DEFAULT_PARAMS['denoise_threshold']:.2f}")
        add_slider_row(noise_layout, 'Strength', self.denoise_slider, self.denoise_value)

        right_col.addWidget(noise_group)
        right_col.addStretch()

        # ==============================================================
        # Assemble columns
        # ==============================================================
        columns.addWidget(left_col_widget, 1)
        columns.addWidget(right_col_widget, 1)

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
        dlg.resize(700, 700)
        dlg.exec()

    def _create_help_tab(self) -> QWidget:
        """Create the help/information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        help_text = """
        <h3 style="color: #88aaff;">About This Tool</h3>
        <p>This tool implements astrophotography image optimisation techniques.
        It uses wavelet decomposition and advanced filtering to harmonise the sky background while preserving astronomical details.</p>

        <h4 style="color: #88aaff;">How It Works</h4>
        <ul>
        <li><b>Color & Levels:</b> Lab‑space adjustments for vibrance, saturation and contrast are applied after reconstruction.</li>
        <li><b>Background Intensity:</b> A smooth background is estimated from the luminance channel using a median filter followed by Gaussian blur, then subtracted to isolate stars and nebulae.</li>
        <li><b>Star Protection:</b> A sigma‑based mask protects bright stars and extended objects from being flattened during background subtraction.</li>
        <li><b>Wavelet Decomposition & Denoising:</b> The residual luminance is decomposed into wavelet levels; low‑frequency detail is suppressed by denoising and high‑frequency detail is preserved.</li>
        <li><b>Dark Mask:</b> An optional radial opacity mask softens the image edges and reduces stray light.</li>
        <li><b>Blending:</b> The processed image is blended with the original using the <i>Intensity</i> slider, allowing fine control over how much background removal is visible.</li>
        <li><b>Output:</b> Final image is clipped to [0, 1] and displayed.</li>
        </ul>

        <h4 style="color: #88aaff;">Usage Tips</h4>
        <ul>
        <li>Start with the default settings; most images look good out of the box.</li>
        <li>Use the <i>Background Intensity</i> slider to decide how aggressively you want to suppress the sky background (0 = no change, 1 = full removal).</li>
        <li>Increase <i>Star Protection</i> (sigma threshold) if bright stars or nebulae appear washed out.</li>
        <li>If the image is noisy, raise <i>Denoising Strength</i> or increase wavelet levels to capture finer detail.</li>
        <li>The <i>Dark Mask Opacity</i> is optional; use it sparingly to soften halo artifacts near the frame edges.</li>
        <li>The <i>Blending Intensity</i> slider lets you mix the processed result with the original; useful for subtle enhancements.</li>
        <li>Use live preview to tweak parameters interactively; full‑resolution processing is triggered by the “Process Full Image” button.</li>
        <li>After processing, click “Save Result” to write the image back into SIRIL or as a TIFF file.</li>
        </ul>

        <h4 style="color: #88aaff;">Performance</h4>
        <p>Processing time depends on image size and parameters. The live preview uses a downscaled version for real‑time feedback, while full resolution processing is applied when you click “Process Full Image”.</p>

        <h4 style="color: #88aaff;">Credits</h4>
        <p>Developed for SIRIL.<br>(c) G. Trainar (2026)</p>
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
        self.btn_load.clicked.connect(self._load_image)
        self.btn_reset.clicked.connect(self._reset_parameters)
        self.btn_process.clicked.connect(self._process_full_image)
        self.btn_save.clicked.connect(self._save_result)
        self.help_btn.clicked.connect(self._show_help)
        self.btn_close.clicked.connect(self.close)

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
        self.black_slider.valueChanged.connect(
            lambda val: self._on_param_change('black_point', val / 100.0))
        self.white_slider.valueChanged.connect(
            lambda val: self._on_param_change('white_point', val / 100.0))
        self.stretch_slider.valueChanged.connect(
            lambda val: self._on_param_change('stretch_intensity', val / 100.0))
        self.vibrance_slider.valueChanged.connect(
            lambda val: self._on_param_change('vibrance', val / 100.0))
        self.saturation_slider.valueChanged.connect(
            lambda val: self._on_param_change('saturation', val / 100.0))
        self.contrast_slider.valueChanged.connect(
            lambda val: self._on_param_change('contrast', val / 100.0))
        self.mask_slider.valueChanged.connect(
            lambda val: self._on_param_change('dark_mask_opacity', val / 100.0))

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
        elif param_name == 'black_point':
            self.black_value.setText(f"{value:.2f}")
        elif param_name == 'white_point':
            self.white_value.setText(f"{value:.2f}")
        elif param_name == 'stretch_intensity':
            self.stretch_value.setText(f"{value:.2f}")
        elif param_name == 'vibrance':
            self.vibrance_value.setText(f"{value:.2f}")
        elif param_name == 'saturation':
            self.saturation_value.setText(f"{value:.2f}")
        elif param_name == 'contrast':
            self.contrast_value.setText(f"{value:.2f}")

        self._preview_timer.start(200)

    def _reset_parameters(self):
        """Reset all sliders and engine parameters to their default values."""
        # Block signals while resetting to avoid triggering many preview redraws
        sliders = [
            self.black_slider, self.white_slider, self.stretch_slider,
            self.vibrance_slider, self.saturation_slider, self.contrast_slider,
            self.mask_slider, self.intensity_slider, self.sigma_slider,
            self.median_slider, self.wavelet_slider, self.denoise_slider,
        ]
        for sl in sliders:
            sl.blockSignals(True)

        self.black_slider.setValue(int(DEFAULT_PARAMS['black_point'] * 100))
        self.white_slider.setValue(int(DEFAULT_PARAMS['white_point'] * 100))
        self.stretch_slider.setValue(int(DEFAULT_PARAMS['stretch_intensity'] * 100))
        self.vibrance_slider.setValue(int(DEFAULT_PARAMS['vibrance'] * 100))
        self.saturation_slider.setValue(int(DEFAULT_PARAMS['saturation'] * 100))
        self.contrast_slider.setValue(int(DEFAULT_PARAMS['contrast'] * 100))
        self.mask_slider.setValue(int(DEFAULT_PARAMS['dark_mask_opacity'] * 100))
        self.intensity_slider.setValue(int(DEFAULT_PARAMS['intensity'] * 100))
        self.sigma_slider.setValue(int(DEFAULT_PARAMS['sigma_threshold'] * 10))
        self.median_slider.setValue(DEFAULT_PARAMS['median_kernel_size'])
        self.wavelet_slider.setValue(DEFAULT_PARAMS['wavelet_levels'])
        self.denoise_slider.setValue(int(DEFAULT_PARAMS['denoise_threshold'] * 100))

        for sl in sliders:
            sl.blockSignals(False)

        # Sync value labels
        self.black_value.setText(f"{DEFAULT_PARAMS['black_point']:.2f}")
        self.white_value.setText(f"{DEFAULT_PARAMS['white_point']:.2f}")
        self.stretch_value.setText(f"{DEFAULT_PARAMS['stretch_intensity']:.2f}")
        self.vibrance_value.setText(f"{DEFAULT_PARAMS['vibrance']:.2f}")
        self.saturation_value.setText(f"{DEFAULT_PARAMS['saturation']:.2f}")
        self.contrast_value.setText(f"{DEFAULT_PARAMS['contrast']:.2f}")
        self.mask_value.setText(f"{DEFAULT_PARAMS['dark_mask_opacity']:.2f}")
        self.intensity_value.setText(f"{DEFAULT_PARAMS['intensity']:.2f}")
        self.sigma_value.setText(f"{DEFAULT_PARAMS['sigma_threshold']:.1f}")
        self.median_value.setText(str(DEFAULT_PARAMS['median_kernel_size']))
        self.wavelet_value.setText(str(DEFAULT_PARAMS['wavelet_levels']))
        self.denoise_value.setText(f"{DEFAULT_PARAMS['denoise_threshold']:.2f}")

        # Reset engine params and trigger a single preview refresh
        self.engine.params = DEFAULT_PARAMS.copy()
        self.status_label.setText("Parameters reset to defaults")
        self.status_label.setStyleSheet("color: #88aaff; font-size: 10pt; background: transparent;")
        if self.current_image is not None:
            self._start_preview()

    # Dark-mask debounce helpers
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

    # Resize-timer helpers
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
        """Re-scale the preview image to fit the current size of the
        preview widget. Called after a resize event and after each
        preview update."""
        if self.original_image is None:
            return
        try:
            if self.show_original:
                img_to_show = np.flipud(self.original_image)
            elif self.engine.last_processed is not None:
                orig_h, orig_w = self.original_image.shape[:2]
                img_to_show = cv2.resize(
                    np.flipud(self.engine.last_processed),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_AREA)
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

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    def _load_image(self):
        """Load button handler.

        - If no image is loaded yet: try SIRIL first, fall back to file dialog.
        - If an image is already loaded: open a file dialog starting in the
          same directory as the current file so the user can switch images
          without losing their place in the file system.
        """
        if self.current_image is not None and self.original_file_path:
            # An image is already open — go straight to a file dialog rooted
            # in the working directory of the current file.
            start_dir = os.path.dirname(self.original_file_path)
            self._load_from_file(start_dir=start_dir)
        else:
            # First load: try SIRIL, then fall back to file dialog.
            if not self._load_from_siril():
                self._load_from_file()

    def _load_from_siril(self) -> bool:
        """Try to load the image currently open in SIRIL.

        Returns True on success, False if no image is available or an error
        occurs (in which case the caller can fall back to a file dialog).
        """
        try:
            current_fname = self.siril.get_image_filename()
            if not current_fname:
                return False

            def is_tiff(fn):
                return fn.lower().endswith(('.tif', '.tiff'))

            if not is_tiff(current_fname):
                try:
                    dir_name, base = os.path.split(current_fname)
                    new_base = os.path.splitext(base)[0]
                    new_path = os.path.join(dir_name, new_base)
                    self.siril.cmd(f'savetif32 {shlex.quote(new_path)}')
                    self.siril.cmd(f'load {shlex.quote(new_path)}')
                    current_fname = new_path
                except Exception as e:
                    raise RuntimeError(f"Could not convert to 32-bit TIFF: {e}")

            img = self.siril.get_image()
            if img is None:
                return False

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

            self._set_loaded_image(image_data, current_fname)
            self.status_label.setText(f"Image loaded from SIRIL: {os.path.basename(current_fname)}")
            self.status_label.setStyleSheet("color: #88ff88; font-size: 10pt; background: transparent;")
            return True

        except Exception as e:
            print(f"SIRIL auto-load skipped: {e}")
            return False

    def _load_from_file(self, start_dir: str = ""):
        """Open a file dialog and load the selected image."""
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilter("Images (*.tif *.tiff *.fit *.fits *.png *.jpg)")
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            if start_dir:
                file_dialog.setDirectory(start_dir)

            if file_dialog.exec() != QFileDialog.DialogCode.Accepted:
                return

            file_path = file_dialog.selectedFiles()[0]
            self.siril.cmd(f'load {shlex.quote(file_path)}')
            img = self.siril.get_image()
            if img is None:
                raise RuntimeError(f"Failed to load image: {file_path}")

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

            self._set_loaded_image(image_data, file_path)
            self.status_label.setText(f"Image loaded: {os.path.basename(file_path)}")
            self.status_label.setStyleSheet("color: #88ff88; font-size: 10pt; background: transparent;")

        except Exception as e:
            self.status_label.setText(f"Error loading image: {e}")
            self.status_label.setStyleSheet("color: #ff8888; font-size: 10pt; background: transparent;")
            print(f"Load image error: {e}")
            traceback.print_exc()

    def _set_loaded_image(self, image_data: np.ndarray, file_path: str):
        """Common finalisation after any successful image load."""
        self.original_image = image_data.copy()
        self.current_image = image_data.copy()
        self.original_file_path = file_path

        # Record full-res dimensions so _estimate_background can scale its
        # kernel and sigma consistently for both preview and full-res renders.
        self.engine.set_reference_image(image_data)

        self._display_image(np.flipud(self.original_image), fit=True)
        self._first_preview_done = False
        self.show_original = False

        self._start_preview()
        self.btn_process.setEnabled(True)
        self.btn_process.setToolTip("Apply all settings to the full-resolution image\n(preview runs on a downscaled version)")
        self.btn_reset.setEnabled(True)
        self.btn_load.setToolTip("Open a file from the current working directory")

    def _start_preview(self):
        """Start or update the preview worker with the current image."""
        if self.current_image is None:
            return

        if hasattr(self, "preview_worker") and self.preview_worker.isRunning():
            self.preview_worker.request_update()  # ← was: set_image + return
            return

        self.preview_worker = PreviewWorker(self.engine, self.preview_label)
        self.preview_worker.preview_ready.connect(
            self._update_preview, Qt.ConnectionType.QueuedConnection)
        self.preview_worker.processing_time.connect(self._handle_processing_time)
        self.preview_worker.set_image(self.current_image)
        self.preview_worker.start()

    def _update_preview(self, preview_data):
        if self._is_resizing:
            return
        try:
            self.engine.last_processed = preview_data
            if self.show_original:
                img_to_show = np.flipud(self.original_image)
            else:
                # Upscale the downscaled preview result back to original image
                # dimensions so that setBaseImage always receives images of the
                # same pixel size.  This keeps zoom_factor and pan offsets stable
                # across slider updates (no drift from mismatched image sizes).
                orig_h, orig_w = self.original_image.shape[:2]
                img_to_show = cv2.resize(
                    np.flipud(preview_data),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR)
            self._display_image(img_to_show, fit=False)
            self.original_indicator.setVisible(self.show_original)
            t = getattr(self, 'last_processing_time', 0.0)
            if t > 1.0:
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
            self.status_label.setStyleSheet("color: #ffcc00; font-size: 10pt; background: transparent;")
            self.app.processEvents()

            result = self.engine.process_image(self.current_image)
            self.current_image = result
            self.show_original = False
            self.engine.last_processed = result  
            self._display_image(np.flipud(result), fit=False)

            self.status_label.setText("Processing complete!")
            self.status_label.setStyleSheet("color: #88ff88; font-size: 10pt; background: transparent;")
            self.btn_save.setEnabled(True)
            self.btn_save.setToolTip("Save the processed image back to SIRIL or as a TIFF file")

        except Exception as e:
            self.status_label.setText(f"Processing failed: {e}")
            self.status_label.setStyleSheet("color: #ff8888; font-size: 10pt; background: transparent;")
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
                self.status_label.setStyleSheet("color: #88ff88; font-size: 10pt; background: transparent;")

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
                    self.status_label.setStyleSheet("color: #88ff88; font-size: 10pt; background: transparent;")

        except Exception as e:
            self.status_label.setText(f"Save failed: {e}")
            self.status_label.setStyleSheet("color: #ff8888; font-size: 10pt; background: transparent;")
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
        gui.setMinimumHeight(800)
        gui.resize(1400, 900)
        gui.show()
        sys.exit(app.exec())

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
