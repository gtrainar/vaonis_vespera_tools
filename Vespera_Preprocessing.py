##############################################
# Vespera — Preprocessing
# Automated Stacking for Alt‑Az Mounts
##############################################
# (c) 2025 G. Trainar - MIT License
# Vespera Preprocessing
# Version 1.1.0
#
# Credits / Origin
# ----------------
#   • Based on Siril's OSC_Preprocessing_BayerDrizzle.ssf
#   • Optimized for Vaonis Vespera II and Pro telescopes
#   • Handles single dark frame capture (Expert Mode)
##############################################

"""
Overview
--------
Full‑featured preprocessing script for Vaonis Vespera astrophotography data.
Designed to handle the unique characteristics of alt‑az mounted smart telescopes
including different sky conditions.

Features
--------
• Bayer Drizzle: Handles field rotation from alt‑az tracking without grid artifacts
• Single Dark Support: Automatically detects and handles 1 or multiple dark frames
• Sky Quality Presets: Optimized settings for dark to urban skies
• Auto Cleanup: Removes all temporary files after successful processing

Compatibility
-------------
• Siril 1.4+
• Python 3.10+ (via sirilpy)
• Dependencies: sirilpy, PyQt6

License
-------
Released under MIT License.
"""

import sys
import os
import glob
import shutil
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

s.ensure_installed("PyQt6", "astropy")

from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)
from astropy.io import fits

# ---------------------------------------------------------------------------
# Version & Changelog
# ---------------------------------------------------------------------------
VERSION = "1.2.0"

CHANGELOG = """
Version 1.2.0 (2026-02)
- Post-Stacking Options (SPCC, Autostretch)
- GUI update
- Code refactoring

Version 1.1.0 (2026-02)
- Batch Processing for disk optimization
- File Dialog for Working Directory

Version 1.0.0 (2026‑01)
• ProcessingProgress constants for standardized progress tracking
• Implemented feathering option (0‑100px) to reduce stacking artifacts
• Added two‑pass registration with framing for improved field rotation handling
• Enhanced logging system with color‑coded messages (red/green/blue/salmon)
• Improved error handling and validation throughout the processing pipeline
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class ProcessingProgress:
    """Standardised progress percentages for each pipeline stage."""
    CLEANUP          = 5
    DARK_PROCESSING  = 10
    LIGHT_CONVERSION = 20
    CALIBRATION      = 30
    REGISTRATION     = 50
    STACKING         = 75
    FINALIZATION     = 88
    COMPLETE         = 100


# Sky quality presets keyed by Bortle description
SKY_PRESETS: Dict[str, Dict[str, Any]] = {
    "Bortle 1-2 (Excellent Dark)": {
        "description": "Remote dark sites, minimal light pollution",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
    },
    "Bortle 3-4 (Rural)": {
        "description": "Rural areas, some light domes on horizon",
        "sigma_low": 3.0,
        "sigma_high": 3.0,
    },
    "Bortle 5-6 (Suburban)": {
        "description": "Suburban skies, noticeable light pollution",
        "sigma_low": 2.5,
        "sigma_high": 3.0,
    },
    "Bortle 7-8 (Urban)": {
        "description": "City skies, heavy light pollution",
        "sigma_low": 2.0,
        "sigma_high": 2.5,
    },
}

# Stacking methods with technical metadata
STACKING_METHODS: Dict[str, Dict[str, Any]] = {
    "Bayer Drizzle (Recommended)": {
        "description": "Best for field rotation, gaussian kernel for smooth CFA",
        "tooltip": (
            "Uses Gaussian drizzle kernel with area‑based interpolation.\n\n"
            "• Gaussian kernel: Produces smooth, centrally‑peaked PSFs\n"
            "• Area interpolation: Reduces moiré patterns from field rotation\n"
            "• Best choice for typical Vespera sessions with 10‑15° rotation\n\n"
            "Technical: scale=1.0, pixfrac=1.0, kernel=gaussian, interp=area"
        ),
        "use_drizzle": True,
        "drizzle_scale": 1.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "gaussian",
        "interp": "area",
        "feather_px": 0,
    },
    "Bayer Drizzle (Square)": {
        "description": "Classic drizzle kernel, mathematically flux‑preserving",
        "tooltip": (
            "Uses classic square drizzle kernel (original HST algorithm).\n\n"
            "• Square kernel: Mathematically flux‑preserving by construction\n"
            "• May show subtle grid patterns with significant field rotation\n"
            "• Better for photometry applications\n\n"
            "Technical: scale=1.0, pixfrac=1.0, kernel=square, interp=area"
        ),
        "use_drizzle": True,
        "drizzle_scale": 1.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "square",
        "interp": "area",
        "feather_px": 0,
    },
    "Bayer Drizzle (Nearest)": {
        "description": "Nearest‑neighbor interpolation to minimize moiré patterns",
        "tooltip": (
            "Uses nearest‑neighbor interpolation to eliminate moiré.\n\n"
            "• Nearest interpolation: No interpolation artifacts at CFA boundaries\n"
            "• May appear slightly blocky at pixel level\n"
            "• Try this if other methods show checkerboard patterns\n\n"
            "Technical: scale=1.0, pixfrac=1.0, kernel=gaussian, interp=nearest"
        ),
        "use_drizzle": True,
        "drizzle_scale": 1.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "gaussian",
        "interp": "nearest",
        "feather_px": 0,
    },
    "Standard Registration": {
        "description": "Faster processing, good for short sessions with minimal rotation",
        "tooltip": (
            "Standard debayer‑then‑register workflow (no drizzle).\n\n"
            "• Faster processing, lower memory usage\n"
            "• Works well for sessions under 30 minutes\n"
            "• May show field rotation artifacts at image edges\n"
            "• Not recommended for sessions with >5° total rotation"
        ),
        "use_drizzle": False,
        "feather_px": 0,
    },
    "Drizzle 2x Upscale": {
        "description": "Doubles resolution, requires many well‑dithered frames (50+)",
        "tooltip": (
            "Upscales to 2x resolution using drizzle algorithm.\n\n"
            "• Requires 50+ frames with good sub‑pixel dithering\n"
            "• Output will be 7072×7072 pixels (vs 3536×3536)\n"
            "• Uses square kernel (only valid choice for scale>1)\n"
            "• Significantly increased processing time and file sizes\n\n"
            "Note: Lanczos kernels cannot be used with scale>1.0\n"
            "Technical: scale=2.0, pixfrac=1.0, kernel=square, interp=area"
        ),
        "use_drizzle": True,
        "drizzle_scale": 2.0,
        "drizzle_pixfrac": 1.0,
        "drizzle_kernel": "square",
        "interp": "area",
        "feather_px": 0,
    },
}

# Telescope specs keyed by model name
TELESCOPES: Dict[str, Dict[str, Any]] = {
    "Vespera II":  {"focal_length_mm": 250.0, "pixel_size_um": 2.9,  "spcc_sensor": "Sony IMX585"},
    "Vespera Pro": {"focal_length_mm": 250.0, "pixel_size_um": 2.00, "spcc_sensor": "Sony IMX676"},
}

# Qt colour mapping for LogColor values
_LOG_COLOR_MAP = {
    LogColor.RED:    Qt.GlobalColor.red,
    LogColor.GREEN:  Qt.GlobalColor.darkGreen,
    LogColor.BLUE:   Qt.GlobalColor.cyan,
    LogColor.SALMON: Qt.GlobalColor.magenta,
}

# ---------------------------------------------------------------------------
# Dark stylesheet
# ---------------------------------------------------------------------------
DARK_STYLESHEET = """
QDialog { background-color: #2b2b2b; color: #e0e0e0; }
QTabWidget::pane { border: 1px solid #444444; background-color: #2b2b2b; }
QTabBar::tab {
    background-color: #3c3c3c; color: #aaaaaa;
    padding: 8px 16px; border: 1px solid #444444;
    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; }
QTabBar::tab:hover    { background-color: #444444; }
QGroupBox {
    border: 1px solid #444444; margin-top: 12px; font-weight: bold;
    border-radius: 4px; padding-top: 8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #88aaff; }
QLabel             { color: #cccccc; font-size: 10pt; }
QLabel#title       { color: #88aaff; font-size: 14pt; font-weight: bold; }
QLabel#subtitle    { color: #888888; font-size: 9pt; }
QLabel#status      { color: #ffcc00; font-size: 10pt; }
QLabel#error       { color: #ff8888; }
QLabel#info        { color: #88aaff; font-size: 9pt; }
QComboBox {
    background-color: #3c3c4c; color: #ffffff; border: 1px solid #555555;
    border-radius: 4px; padding: 5px 10px; width: 176px;
}
QComboBox:hover { border-color: #88aaff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #aaaaaa;
}
QComboBox QAbstractItemView {
    background-color: #3c3c4c; color: #ffffff;
    selection-background-color: #285299; border: 1px solid #555555;
}
QCheckBox { color: #cccccc; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #666666;
    background: #3c3c4c; border-radius: 3px;
}
QCheckBox::indicator:checked  { background-color: #285299; border: 1px solid #88aaff; }
QCheckBox::indicator:hover    { border-color: #88aaff; }
QSpinBox, QDoubleSpinBox {
    background-color: #3c3c4c; color: #ffffff; border: 1px solid #555555;
    border-radius: 4px; padding: 4px; width: 180px;
}
QProgressBar {
    border: 1px solid #555555; border-radius: 4px; background-color: #3c3c4c;
    text-align: center; color: #ffffff; min-height: 20px;
}
QProgressBar::chunk { background-color: #285299; border-radius: 3px; }
QPushButton {
    background-color: #444444; color: #dddddd; border: 1px solid #666666;
    border-radius: 4px; padding: 8px 20px; font-weight: bold; min-width: 100px;
}
QPushButton:hover    { background-color: #555555; border-color: #777777; }
QPushButton:pressed  { background-color: #333333; }
QPushButton:disabled { background-color: #333333; color: #666666; }
QPushButton#start          { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#start:hover    { background-color: #3366bb; }
QPushButton#start:disabled { background-color: #1a1a2e; color: #555555; }
QTextEdit {
    background-color: #1e1e1e; color: #aaaaaa; border: 1px solid #444444;
    border-radius: 4px; font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
    font-size: 9pt; padding: 5px;
}
QFrame#separator { background-color: #444444; min-height: 1px; max-height: 1px; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_fits_in(folder: str) -> int:
    """Return the number of FITS files in *folder* (all common extensions)."""
    return sum(
        len(glob.glob(os.path.join(folder, ext)))
        for ext in ("*.fit", "*.fits", "*.FIT", "*.FITS")
    )


def append_colored_text(text_edit: QTextEdit, msg: str, color: Optional[LogColor]) -> None:
    """Append *msg* to *text_edit* using the colour that corresponds to *color*."""
    cursor = text_edit.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    text_edit.setTextCursor(cursor)
    qt_color = _LOG_COLOR_MAP.get(color, Qt.GlobalColor.lightGray)
    text_edit.setTextColor(qt_color)
    text_edit.append(msg)
    text_edit.setTextColor(Qt.GlobalColor.lightGray)  # reset


# ---------------------------------------------------------------------------
# Disk‑usage monitor thread
# ---------------------------------------------------------------------------
class DiskUsageThread(QThread):
    """Background thread that logs free/total disk space every *interval_sec* seconds."""

    def __init__(
        self,
        log_file: Path,
        workdir: Path,
        interval_sec: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.log_file = log_file
        self.workdir = workdir
        self.interval_sec = interval_sec
        self._running = True

    def run(self) -> None:
        with open(self.log_file, "a", encoding="utf-8") as fh:
            while self._running:
                try:
                    total, _used, free = shutil.disk_usage(self.log_file.parent)
                    dir_size = sum(
                        f.stat().st_size
                        for f in self.workdir.rglob("*")
                        if f.is_file()
                    )
                    fh.write(f"{datetime.now().isoformat()},{free},{total},{dir_size}\n")
                except Exception as exc:
                    fh.write(f"{datetime.now().isoformat()},ERROR,{exc}\n")
                time.sleep(self.interval_sec)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Plate solver
# ---------------------------------------------------------------------------
class VesperaPlateSolver:
    """
    Plate‑solving helper for Vespera 16‑bit TIFF images.
    """

    # Regex to extract six numeric groups from a SIMBAD ICRS coordinate line
    _ICRS_RE = re.compile(
        r"(\d+)\s+(\d+)\s+([\d.]+)\s+([+-]?\d+)\s+(\d+)\s+([\d.]+)"
    )

    def __init__(
        self,
        siril_interface: Any,
        filename: Optional[str] = None,
        focal_length_mm: float = 250.0,
        pixel_size_um: float = 2.00,
    ) -> None:
        self.siril = siril_interface
        self.filename = filename
        self.focal_length_mm = focal_length_mm
        self.pixel_size_um = pixel_size_um
        self.dso_name: Optional[str] = None
        self.applied_coordinates: Optional[tuple] = None

        if filename:
            self._extract_dso_name()

    # ------------------------------------------------------------------
    def _extract_dso_name(self) -> None:
        """Read the OBJECT keyword from the FITS header."""
        try:
            with fits.open(self.filename) as hdulist:
                self.dso_name = str(hdulist[0].header.get("OBJECT", "")).strip() or None
        except Exception as exc:
            self.siril.log(f"DSO extraction error: {exc}", LogColor.SALMON)

    # ------------------------------------------------------------------
    def plate_solve(self, ra_deg: Optional[float] = None, dec_deg: Optional[float] = None) -> bool:
        """
        Run Siril's platesolve command.
        """
        try:
            coords_arg = (
                f" {ra_deg:.6f},{dec_deg:.6f}"
                if ra_deg is not None and dec_deg is not None
                else ""
            )
            cmd = (
                f"platesolve{coords_arg}"
                f" -focal={self.focal_length_mm}"
                f" -pixelsize={self.pixel_size_um}"
            )
            self.siril.cmd(cmd)
            return True
        except Exception as exc:
            self.siril.log(f"Plate solve error: {exc}", LogColor.SALMON)
            return False

    # ------------------------------------------------------------------
    def _query_simbad_coordinates(self, dso_name: str) -> Optional[tuple]:
        """
        Query the SIMBAD TAP service for ICRS coordinates of *dso_name*.
        """
        import urllib.parse
        import urllib.request

        params = {"output.format": "ASCII", "Ident": dso_name}
        url = f"https://simbad.cds.unistra.fr/simbad/sim-id?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode("utf-8")
        except Exception as exc:
            self.siril.log(f"SIMBAD query error: {exc}", LogColor.SALMON)
            return None

        for line in data.splitlines():
            if not line.startswith("Coordinates(ICRS,ep=J2000,eq=2000):"):
                continue

            _, coord_part = line.split(":", 1)
            match = self._ICRS_RE.search(coord_part)
            if not match:
                self.siril.log(
                    "Could not parse ICRS coordinates from SIMBAD response",
                    LogColor.SALMON,
                )
                return None

            ra_h, ra_m, ra_s, dec_d, dec_m, dec_s = match.groups()
            ra_deg  = 15.0 * (float(ra_h) + float(ra_m) / 60.0 + float(ra_s) / 3600.0)
            sign    = -1 if dec_d.strip().startswith("-") else 1
            dec_deg = sign * (abs(float(dec_d)) + float(dec_m) / 60.0 + float(dec_s) / 3600.0)

            self.siril.log(
                f"SIMBAD coordinates for {dso_name}: RA={ra_deg:.6f} DEC={dec_deg:.6f}",
                LogColor.BLUE,
            )
            return ra_deg, dec_deg

        self.siril.log(
            f"SIMBAD did not return coordinates for {dso_name}", LogColor.SALMON
        )
        return None


# ---------------------------------------------------------------------------
# Processing thread
# ---------------------------------------------------------------------------
class ProcessingThread(QThread):
    """Background thread that runs the full Siril preprocessing pipeline."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log      = pyqtSignal(str)

    def __init__(
        self,
        siril: Any,
        workdir: str,
        settings: Dict[str, Any],
        folder_structure: str,
    ) -> None:
        super().__init__()
        self.siril            = siril
        self.workdir          = workdir
        self.settings         = settings
        self.folder_structure = folder_structure   # 'native' | 'organized'
        self.log_area: Optional[QTextEdit] = None
        self.console_messages: List[str]   = []
        self.light_seq_name   = "light"
        self.final_filename   = ""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        """Emit *msg* to the GUI log area and to Siril's console."""
        if self.log_area:
            append_colored_text(self.log_area, msg, color)

        try:
            self.siril.log(msg, color=color) if color else self.siril.log(msg)
        except Exception as exc:
            # Avoid recursive logging errors
            self.console_messages.append(f"Logging error: {exc}")

        self.console_messages.append(msg)

    # ------------------------------------------------------------------
    # Siril command wrapper
    # ------------------------------------------------------------------
    def _run(self, *cmd: str) -> bool:
        """Execute a Siril command and return True on success."""
        try:
            self.siril.cmd(*cmd)
            return True
        except (RuntimeError, Exception) as exc:
            self._log(f"Error in '{' '.join(cmd)}': {exc}", LogColor.RED)
            return False

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    @property
    def _process_dir(self) -> str:
        return os.path.normpath(os.path.join(self.workdir, "process"))

    @property
    def _masters_dir(self) -> str:
        return os.path.normpath(os.path.join(self.workdir, "masters"))

    def _create_final_stack_dir(self) -> Path:
        """Return (and create if needed) the directory for batch result FITS."""
        final_dir = Path(self.workdir, "final_stack")
        try:
            final_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Could not create final stack dir {final_dir}: {exc}")
        return final_dir

    # ------------------------------------------------------------------
    # FITS file counters / movers
    # ------------------------------------------------------------------
    def _count_fits(self, folder: str) -> int:
        return count_fits_in(folder)

    def _move_tiff_to_reference(self, lights_dir: str) -> int:
        """Move any TIFF reference images found in *lights_dir* to ``reference/``."""
        tiff_files = [
            f
            for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
            for f in glob.glob(os.path.join(lights_dir, ext))
        ]
        if not tiff_files:
            return 0

        reference_dir = os.path.join(self.workdir, "reference")
        try:
            os.makedirs(reference_dir, exist_ok=True)
        except Exception as exc:
            self._log(f"Could not create reference dir: {exc}", LogColor.RED)
            return 0

        moved = 0
        for src in tiff_files:
            try:
                dest = os.path.join(reference_dir, os.path.basename(src))
                shutil.move(src, dest)
                self._log(f"Moved reference image: {os.path.basename(src)} → reference/", LogColor.SALMON)
                moved += 1
            except Exception as exc:
                self._log(f"Warning: Could not move {src}: {exc}", LogColor.SALMON)
        return moved

    # ------------------------------------------------------------------
    # Temporary file cleanup
    # ------------------------------------------------------------------
    def _cleanup_folder(self, folder) -> int:
        """
        Remove FITS, sequence, and temporary sub‑directories from *folder*.
        """
        folder = str(folder)
        if not os.path.exists(folder):
            return 0

        count = 0
        cleanup_patterns = ("*.fit", "*.fits", "*.FIT", "*.FITS", "*.seq", "*conversion.txt")
        cleanup_subdirs  = ("cache", "drizztmp", "other")

        for pattern in cleanup_patterns:
            for filepath in glob.glob(os.path.join(folder, pattern)):
                try:
                    os.remove(filepath)
                    count += 1
                except OSError as exc:
                    self._log(f"Warning: Could not remove {filepath}: {exc}", LogColor.SALMON)

        for subdir_name in cleanup_subdirs:
            subdir_path = os.path.join(folder, subdir_name)
            if os.path.isdir(subdir_path):
                try:
                    shutil.rmtree(subdir_path)
                    count += 1
                except OSError as exc:
                    self._log(f"Warning: Could not remove {subdir_path}: {exc}", LogColor.SALMON)

        final_stack_dir = Path(folder).parent / "final_stack"
        if final_stack_dir.is_dir():
            try:
                shutil.rmtree(final_stack_dir)
                count += 1
                self._log(f"Removed final stack directory: {final_stack_dir}", LogColor.BLUE)
            except Exception as exc:
                self._log(f"Could not delete final_stack: {exc}", LogColor.RED)

        return count

    # ------------------------------------------------------------------
    # Telescope auto‑detection
    # ------------------------------------------------------------------
    def _set_telescope_from_fits(self) -> None:
        """Detect telescope model from first FITS header and update settings."""
        lights_dir = os.path.join(self.workdir, "lights")
        fits_files = [
            f for f in os.listdir(lights_dir)
            if f.lower().endswith((".fits", ".fit", ".fits.fz", ".fit.fz"))
        ]
        if not fits_files:
            return

        first_file = os.path.join(lights_dir, fits_files[0])
        try:
            with fits.open(first_file) as hdul:
                naxis1 = hdul[0].header.get("NAXIS1", 0)
                naxis2 = hdul[0].header.get("NAXIS2", 0)
        except Exception as exc:
            self.siril.log(f"Error reading telescope from FITS: {exc}", LogColor.SALMON)
            return

        if naxis1 == 3536 and naxis2 == 3536:
            model = "Vespera Pro"
        elif naxis1 == 3840 and naxis2 == 2160:
            model = "Vespera II"
        else:
            self.siril.log("Couldn't find telescope info, using defaults", LogColor.BLUE)
            return

        self.settings.update(TELESCOPES[model])
        self._log(f"Set telescope to {model} from FITS header", LogColor.BLUE)

    # ------------------------------------------------------------------
    # Pipeline stages: calibrate → register → stack
    # ------------------------------------------------------------------
    def _calibrate(self, seq_name: str, stack_method: Dict[str, Any]) -> None:
        master_dark = (
            "../../../masters/dark_stacked"
            if self.settings.get("batch_enabled")
            else "../masters/dark_stacked"
        )
        cmd = ["calibrate", seq_name, f"-dark={master_dark}", "-cc=dark", "-cfa"]

        if stack_method.get("use_drizzle"):
            cmd.append("-equalize_cfa")
        else:
            cmd.extend(["-debayer", "-equalize_cfa"])

        if not self._run(*cmd):
            raise RuntimeError("Calibration failed")

    # ------------------------------------------------------------------
    def _register(self, seq_name: str, stack_method: Dict[str, Any]) -> None:
        use_drizzle   = stack_method.get("use_drizzle", False)
        drizzle_scale = stack_method.get("drizzle_scale", 1.0)
        two_pass      = self.settings.get("two_pass", False)

        cmd = ["register", f"pp_{seq_name}"]
        if use_drizzle:
            cmd += [
                "-drizzle",
                f"-scale={drizzle_scale}",
                f"-pixfrac={stack_method.get('drizzle_pixfrac', 1.0)}",
                f"-kernel={stack_method.get('drizzle_kernel', 'square')}",
                f"-interp={stack_method.get('interp', 'area')}",
            ]
        if two_pass and (not use_drizzle or drizzle_scale == 1.0):
            cmd.append("-2pass")

        if not self._run(*cmd):
            raise RuntimeError("Registration failed")

        if not two_pass:
            return

        # Second pass
        if not use_drizzle:
            if not self._run("seqapplyreg", f"pp_{seq_name}", "-framing=max"):
                raise RuntimeError("2‑pass registration failed")
        elif drizzle_scale == 1.0:
            if not self._run(
                "seqapplyreg", f"pp_{seq_name}",
                "-drizzle", "-filter-round=2.5k",
                f"-scale={drizzle_scale}",
                f"-pixfrac={stack_method.get('drizzle_pixfrac', 1.0)}",
                f"-kernel={stack_method.get('drizzle_kernel', 'square')}",
            ):
                raise RuntimeError("2‑pass drizzle registration failed")

    # ------------------------------------------------------------------
    def _stack(
        self,
        seq_name: str,
        stack_method: Dict[str, Any],
        sigma_low: float,
        sigma_high: float,
        output_name: str,
    ) -> None:
        cmd = [
            "stack", f"r_pp_{seq_name}",
            "rej", str(sigma_low), str(sigma_high),
            "-norm=addscale", "-output_norm",
            "-rgb_equal", "-weight=wfwhm",
        ]
        if self.settings.get("feather_enabled") and self.settings.get("feather_px", 0) > 0:
            cmd.append(f"-feather={self.settings['feather_px']}")
        cmd.append(f"-out={output_name}")

        if not self._run(*cmd):
            raise RuntimeError("Stacking failed")

    # ------------------------------------------------------------------
    # Standard (single‑batch) workflow
    # ------------------------------------------------------------------
    def _process_standard(
        self,
        stack_method: Dict[str, Any],
        sigma_low: float,
        sigma_high: float,
        chunk_idx: Optional[int] = None,
        total_chunks: Optional[int] = None,
    ) -> None:
        """Calibrate → (optional BGE) → Register → Stack → Finalize."""

        def emit(stage_rel: float, msg: str) -> None:
            """Emit progress scaled to the current chunk's slice of the bar."""
            if chunk_idx is not None and total_chunks:
                chunk_span = 58.0 / total_chunks
                start      = 30 + (chunk_idx - 1) * chunk_span
                percent    = int(start + (stage_rel / 58.0) * chunk_span)
            else:
                percent = int(stage_rel)
            self.progress.emit(percent, msg)

        def _chunk_label(base: str) -> str:
            return f"{base} chunk {chunk_idx} of {total_chunks}" if chunk_idx else base

        # -- Calibration -------------------------------------------------
        emit(20, _chunk_label("Calibrating..."))
        try:
            self._calibrate(self.light_seq_name, stack_method)
        except Exception as exc:
            self._log(f"Calibration failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)
            return

        # -- Background extraction (optional) ----------------------------
        if self.settings.get("bge"):
            emit(25, _chunk_label("Background Extraction..."))
            try:
                self._run_background_extraction()
            except Exception as exc:
                self._log(f"Background extraction failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)
                return

        # -- Registration ------------------------------------------------
        emit(30, _chunk_label("Registering..."))
        try:
            self._register(self.light_seq_name, stack_method)
        except Exception as exc:
            self._log(f"Registration failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)
            return

        # -- Stacking ----------------------------------------------------
        emit(45, _chunk_label("Stacking..."))
        try:
            self._stack(self.light_seq_name, stack_method, sigma_low, sigma_high, "result")
        except Exception as exc:
            self._log(f"Stacking failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)
            return

        # -- Finalization ------------------------------------------------
        emit(58, _chunk_label("Finalizing..."))
        try:
            self.siril.cmd("load", "result")
            self.siril.cmd("icc_remove")
            try:
                exposure = self.siril.get_image_fits_header("LIVETIME", default="XX")
                self.final_filename = f"result_{exposure}s.fit"
            except Exception:
                self.final_filename = "result_XXXXs.fit"
            self.siril.cmd("save", "../result_$LIVETIME:%d$s")
            self.siril.cmd("cd", "..")
        except Exception as exc:
            self._log(f"Finalization failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)

    # ------------------------------------------------------------------
    # Batch (multi‑chunk) workflow
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_chunks(src: Path, dest_root: Path, batch_size: int) -> None:
        """
        Sort FITS files in *src* by frame number, split them into
        ``batch_size``‑sized sub‑folders.
        """
        frame_re = re.compile(r".*?(\d+)\.(fits?|fit)$", re.IGNORECASE)

        files: List[Path] = sorted(
            (p for p in src.iterdir() if p.is_file() and frame_re.fullmatch(p.name)),
            key=lambda p: int(frame_re.fullmatch(p.name).group(1)),
        )

        chunks = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

        for idx, batch in enumerate(chunks, start=1):
            lights_dir = dest_root / f"light-{idx:03d}" / "lights"
            lights_dir.mkdir(parents=True, exist_ok=True)

            for file_path in batch:
                shutil.move(str(file_path), str(lights_dir / file_path.name))

            # A single‑frame chunk cannot be stacked — duplicate it
            if len(batch) == 1:
                only = lights_dir / batch[0].name
                shutil.copy2(only, lights_dir / f"{batch[0].stem}_dup{batch[0].suffix}")

    # ------------------------------------------------------------------
    def _process_batch_sessions(
        self,
        stack_method: Dict[str, Any],
        sigma_low: float,
        sigma_high: float,
    ) -> None:
        """Split lights into chunks, stack each independently, then combine."""
        batch_size = int(self.settings.get("batch_size", 100))

        # Save and temporarily disable per‑chunk flags
        saved = {k: self.settings[k] for k in ("two_pass", "feather_enabled")}
        self.settings.update({"two_pass": False, "feather_enabled": False})

        # For 2x upscale, use the recommended method for intermediate stacks
        intermediate_method = (
            STACKING_METHODS["Bayer Drizzle (Recommended)"]
            if stack_method.get("use_drizzle") and stack_method.get("drizzle_scale", 1.0) > 1.0
            else stack_method
        )

        try:
            self._prepare_chunks(
                Path(self.workdir, "process"),
                Path(self.workdir, "Temp"),
                batch_size,
            )
        except Exception as exc:
            self._log(f"Could not prepare chunks: {exc}", LogColor.RED)
            return

        session_dirs = sorted(Path(self.workdir, "Temp").glob("light-*"))
        total_chunks = len(session_dirs)
        final_dir    = self._create_final_stack_dir()

        for idx, sess in enumerate(session_dirs, start=1):
            try:
                if not self._run("cd", f'"{sess}/lights"'):
                    continue
                if not self._run("convert", "light", "-out=../process"):
                    continue
                if not self._run("cd", f'"{sess}/process"'):
                    continue

                self.light_seq_name = "light"
                self._process_standard(
                    intermediate_method, sigma_low, sigma_high,
                    chunk_idx=idx, total_chunks=total_chunks,
                )
            except Exception as exc:
                self._log(f"Chunk {idx} failed: {exc}", LogColor.RED)
                continue

            src_result = sess / "process" / "result.fit"
            if src_result.is_file():
                dst_name = f"pp_session_{idx:03d}.fits"
                try:
                    shutil.move(str(src_result), final_dir / dst_name)
                except Exception as exc:
                    self._log(f"Could not move {src_result} to final stack: {exc}", LogColor.SALMON)
            else:
                self._log(f"No result file for {sess.name}", LogColor.SALMON)

            self._cleanup_folder(sess / "process")

        # Restore original flags before final stack
        self.settings.update(saved)

        self.progress.emit(ProcessingProgress.FINALIZATION, "Finalizing...")
        try:
            self._stack_final(stack_method)
        except Exception as exc:
            self._log(f"Final stack failed: {exc}", LogColor.RED)

        temp_dir = Path(self.workdir, "Temp")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                self._log(f"Removed temporary folder: {temp_dir}", LogColor.BLUE)
            except Exception as exc:
                self._log(f"Could not delete Temp folder: {exc}", LogColor.RED)

    # ------------------------------------------------------------------
    def _stack_final(self, stack_method: Dict[str, Any]) -> None:
        """Register and stack all per‑session FITS into a single final image."""
        temp_dir = Path(self.workdir) / "final_stack"

        try:
            self.siril.cmd("cd", f'"{temp_dir.as_posix()}"')
            self.siril.cmd("convert", "pp_final", '-out=./process')
            self.siril.cmd("cd", "process")

            reg_cmd = ["register", "pp_final"]
            if self.settings.get("two_pass"):
                reg_cmd.append("-2pass")
            if not self._run(*reg_cmd):
                self._log("Registration failed – skipping final stack", LogColor.SALMON)
                return

            if self.settings.get("two_pass"):
                if not self._run("seqapplyreg", "pp_final", "-framing=max"):
                    raise RuntimeError("2‑pass registration failed for final stack")

            stack_cmd = [
                "stack", "r_pp_final",
                "rej", "3", "3",
                "-norm=addscale", "-output_norm",
                "-weight=wfwhm",
            ]
            if self.settings.get("feather_enabled") and self.settings.get("feather_px", 0) > 0:
                stack_cmd.append(f"-feather={self.settings['feather_px']}")
            stack_cmd.append("-out=final_stacked_batch.fit")

            if not self._run(*stack_cmd):
                self._log("Stacking failed – final stack incomplete", LogColor.SALMON)
                return

            final_out = Path(self.workdir) / "final_stacked_batch.fit"
            shutil.move(str(temp_dir / "process" / "final_stacked_batch.fit"), final_out)
            self._log(f"Final stacked image written to {final_out}", LogColor.BLUE)
            self.siril.cmd("cd", "../..")

        except Exception as exc:
            self._log(f"Final stack failed: {exc}", LogColor.SALMON)

    # ------------------------------------------------------------------
    # Post‑stacking pipeline
    # ------------------------------------------------------------------
    def _run_poststacking(self) -> None:
        """Plate solve → SPCC → Auto‑stretch, if enabled."""
        total_steps  = max(sum([
            bool(self.settings.get("bge")),
            bool(self.settings.get("spcc")),
            bool(self.settings.get("autostretch")),
        ]), 1)
        current_step = 0

        if self.settings.get("batch_enabled"):
            self.siril.cmd("load", "final_stacked_batch.fit")

        if self.settings.get("spcc"):
            current_step += 1
            self.progress.emit(int(current_step / total_steps * 100), "Plate solving...")
            plate_solve_ok = self._run_plate_solve(
                self.settings["focal_length_mm"], self.settings["pixel_size_um"]
            )
            if plate_solve_ok:
                current_step += 1
                self.progress.emit(int(current_step / total_steps * 100), "Color calibrating...")
                self._run_spcc()
            else:
                self.siril.log("Plate solving failed – skipping SPCC", LogColor.SALMON)

        if self.settings.get("autostretch"):
            current_step += 1
            self.progress.emit(int(current_step / total_steps * 100), "Auto‑stretching...")
            self._run_autostretch()

        self.progress.emit(100, "Complete!")

    # ------------------------------------------------------------------
    def _run_background_extraction(self) -> None:
        """Subtract the background using Siril's RBF method."""
        self.siril.cmd("subsky", "-rbf", "-samples=60", "-tolerance=1.0", "-smooth=0.5")

    # ------------------------------------------------------------------
    def _run_plate_solve(self, focal_length_mm: float, pixel_size_um: float) -> bool:
        """
        Plate‑solve the current image.
        """

        if self.settings.get("stacking_method") == "Drizzle 2x Upscale":
            focal_length_mm *= 2
            self._log(
                f"Drizzle 2× Upscale detected – using effective focal length "
                f"{focal_length_mm:.1f} mm for plate solving",
                LogColor.BLUE,
            )
        
        try:
            rel      = self.siril.get_image_filename()
            abs_path = Path(self.siril.get_siril_wd()) / Path(rel).name
            solver   = VesperaPlateSolver(
                self.siril, str(abs_path),
                focal_length_mm=focal_length_mm,
                pixel_size_um=pixel_size_um,
            )

            ra_deg = dec_deg = None
            if solver.dso_name:
                coords = solver._query_simbad_coordinates(solver.dso_name)
                if coords:
                    solver.applied_coordinates = coords
                    ra_deg, dec_deg = coords
                else:
                    self.siril.log(
                        f"SIMBAD lookup failed for '{solver.dso_name}'", LogColor.SALMON
                    )

            if not solver.applied_coordinates:
                self.siril.log("Cannot plate solve without valid coordinates", LogColor.SALMON)
                return False

            # Pass ra/dec directly on the CLI — this is what Siril actually uses
            success = solver.plate_solve(ra_deg=ra_deg, dec_deg=dec_deg)
            self.siril.log(
                "Plate solving completed!" if success else "Plate solving failed",
                LogColor.GREEN if success else LogColor.SALMON,
            )
            if not success:
                self.siril.log(
                    "Check telescope selection (Vespera II / Pro) and DSO name, then retry.",
                    LogColor.SALMON,
                )
            return success

        except Exception as exc:
            self.siril.log(f"Plate solving error: {exc}", LogColor.RED)
            return False

    # ------------------------------------------------------------------
    def _run_spcc(self) -> None:
        """Run Spectrophotometric Color Correction for the appropriate filter."""
        filter_name = self.settings.get("spcc_filter", "No Filter").strip()
        sensor      = self.settings["spcc_sensor"]
        self.siril.log(f"SPCC filter: {filter_name}", LogColor.BLUE)

        if filter_name == "City Light Pollution":
            cmd = f'spcc "-oscsensor={sensor}" "-oscfilter=Vaonis CLS"'
        elif filter_name == "Dual Band Ha/Oiii":
            cmd = (
                f'spcc "-oscsensor={sensor}" "-narrowband" '
                f'"-rwl=656.3" "-rbw=12" "-gwl=500.7" "-gbw=12" "-bfilter=NoFilter"'
            )
        else:  # No Filter (default)
            cmd = (
                f'spcc "-oscsensor={sensor}" '
                f'"-rfilter=NoFilter" "-gfilter=NoFilter" "-bfilter=NoFilter"'
            )

        self.siril.log(f"Running SPCC: {cmd}", LogColor.BLUE)
        self.siril.cmd(cmd)

    # ------------------------------------------------------------------
    def _run_autostretch(self) -> None:
        """Apply linked auto‑stretch with configured shadow/background parameters."""
        self.siril.cmd(
            "autostretch", "-linked",
            str(self.settings.get("shadowsclip", -2.8)),
            str(self.settings.get("targetbg",    0.25)),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._process()
        except Exception as exc:
            self.finished.emit(False, f"Error: {exc}")

    def _process(self) -> None:
        """Orchestrate the full preprocessing pipeline."""
        sky      = SKY_PRESETS[self.settings["sky_quality"]]
        method   = STACKING_METHODS[self.settings["stacking_method"]]
        sigma_lo = sky["sigma_low"]
        sigma_hi = sky["sigma_high"]

        process_dir = self._process_dir
        masters_dir = self._masters_dir

        # Resolve paths based on folder layout
        if self.folder_structure == "native":
            dark_file  = os.path.join(self.workdir, "img-0001-dark.fits")
            lights_dir = os.path.join(self.workdir, "01-images-initial")
        else:
            darks_dir  = os.path.join(self.workdir, "darks")
            lights_dir = os.path.join(self.workdir, "lights")

        # Validate required paths
        if self.folder_structure == "native":
            for label, path in (("Dark file", dark_file), ("Lights folder", lights_dir)):
                if not os.path.exists(path):
                    self.finished.emit(False, f"{label} not found: {path}")
                    return
        else:
            for label, path in (("Dark folder", darks_dir), ("Light folder", lights_dir)):
                if not os.path.exists(path):
                    self.finished.emit(False, f"{label} not found: {path}")
                    return

        for d in (process_dir, masters_dir):
            os.makedirs(d, exist_ok=True)

        moved = self._move_tiff_to_reference(lights_dir)
        if moved:
            self._log(f"Moved {moved} TIFF reference image(s) to 'reference/'", LogColor.SALMON)

        # Frame counts
        if self.folder_structure == "native":
            num_darks  = 1
            num_lights = len([
                f for f in (
                    glob.glob(os.path.join(lights_dir, "*.fits")) +
                    glob.glob(os.path.join(lights_dir, "*.fit"))
                )
                if "-dark" not in f.lower()
            ])
        else:
            num_darks  = self._count_fits(darks_dir)
            num_lights = self._count_fits(lights_dir)

        for label, count in (("dark", num_darks), ("light", num_lights)):
            if count == 0:
                self.finished.emit(False, f"No {label} frames found")
                return

        self._log(f"Sky Quality: {self.settings['sky_quality']}",    LogColor.BLUE)
        self._log(f"Stacking:    {self.settings['stacking_method']}", LogColor.BLUE)
        self._log(f"Structure:   {self.folder_structure}",            LogColor.BLUE)
        self._log(f"Found {num_darks} dark(s), {num_lights} light(s)", LogColor.BLUE)

        # === CLEANUP ===
        self.progress.emit(ProcessingProgress.CLEANUP, "Cleaning previous files...")
        if self.settings.get("clean_temp"):
            deleted = self._cleanup_folder(process_dir) + self._cleanup_folder(masters_dir)

        # === DARK PROCESSING ===
        self.progress.emit(ProcessingProgress.DARK_PROCESSING, "Processing darks...")
        if self.folder_structure == "native":
            self._log("Single dark → using directly as master", LogColor.BLUE)
            self.siril.cmd("load", f'"{dark_file}"')
            self.siril.cmd("save", "masters/dark_stacked")
        else:
            self.siril.cmd("cd", "darks")
            self.siril.cmd("convert", "dark", "-out=../masters")
            self.siril.cmd("cd", "../masters")
            if num_darks == 1:
                self._log("Single dark → using directly as master", LogColor.BLUE)
                self.siril.cmd("load", "dark_00001")
                self.siril.cmd("save", "dark_stacked")
            else:
                self._log(f"Stacking {num_darks} darks...", LogColor.BLUE)
                self.siril.cmd(
                    "stack", "dark", "rej",
                    str(sigma_lo), str(sigma_hi),
                    "-nonorm", "-out=dark_stacked",
                )

        # === TELESCOPE DETECTION ===
        self._set_telescope_from_fits()

        # === LIGHT CONVERSION ===
        self.progress.emit(ProcessingProgress.LIGHT_CONVERSION, "Converting lights...")
        self.siril.cmd("cd", "01-images-initial" if self.folder_structure == "native" else "../lights")
        self.siril.cmd("convert", "light", "-out=../process")
        self.siril.cmd("cd", "../process")
        self.light_seq_name = "light"

        # === STACKING ===
        if self.settings.get("batch_enabled"):
            self._process_batch_sessions(method, sigma_lo, sigma_hi)
        else:
            self._process_standard(method, sigma_lo, sigma_hi)

        # === POST‑STACKING ===
        self._run_poststacking()

        # === FINAL CLEANUP ===
        if self.settings.get("clean_temp"):
            self.progress.emit(98, "Cleaning up...")
            deleted = self._cleanup_folder(process_dir) + self._cleanup_folder(masters_dir)
            self._log(f"Cleaned {deleted} temp files", LogColor.BLUE)

        self.progress.emit(ProcessingProgress.COMPLETE, "Complete!")
        self.finished.emit(True, "Processing complete!")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class VesperaProGUI(QDialog):
    """Main preprocessing dialog for Vespera observations."""

    def __init__(self, siril: Any, app: QApplication) -> None:
        super().__init__()
        self.siril    = siril
        self.app      = app
        self.worker:      Optional[ProcessingThread] = None
        self.disk_thread: Optional[DiskUsageThread]  = None
        self.qsettings    = QSettings("Vespera", "Preprocessing")
        self.current_settings: Dict[str, Any] = {}
        self.folder_structure: Optional[str]  = None
        self.workdir  = ""

        self.setWindowTitle(f"Vespera — Preprocessing v{VERSION}")
        self.setMinimumSize(550, 1000)
        self.resize(550, 1000)
        self.setStyleSheet(DARK_STYLESHEET)

        self._setup_ui()
        self._load_settings()
        self._check_folders()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        if self.log_area:
            append_colored_text(self.log_area, msg, color)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._create_main_tab(), "Main")
        tabs.addTab(self._create_info_tab(), "Info")
        layout.addWidget(tabs)

        # Progress section
        progress_group  = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress   = QProgressBar()
        self.progress.setRange(0, 100)
        progress_layout.addWidget(self.progress)

        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        progress_layout.addWidget(self.log_area)
        layout.addWidget(progress_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_start = QPushButton("Start Processing")
        self.btn_start.setObjectName("start")
        self.btn_start.clicked.connect(self._start_processing)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    def _create_main_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Folder status
        status_group  = QGroupBox("Folder Status")
        status_layout = QVBoxLayout(status_group)

        wd_row = QHBoxLayout()
        self.lbl_workdir    = QLabel("Working directory: ...")
        self.btn_browse_dir = QPushButton("Browse…")
        self.btn_browse_dir.clicked.connect(self._browse_working_directory)
        self.btn_browse_dir.setStyleSheet("QPushButton { padding: 2px 6px; }")
        wd_row.addWidget(self.lbl_workdir)
        wd_row.addStretch()
        wd_row.addWidget(self.btn_browse_dir)
        status_layout.addLayout(wd_row)

        folder_row = QHBoxLayout()
        self.lbl_darks  = QLabel("Darks: checking...")
        self.lbl_lights = QLabel("Lights: checking...")
        folder_row.addWidget(self.lbl_darks)
        folder_row.addWidget(self.lbl_lights)
        status_layout.addLayout(folder_row)

        self.lbl_structure = QLabel("")
        self.lbl_structure.setObjectName("info")
        self.lbl_structure.setWordWrap(True)
        status_layout.addWidget(self.lbl_structure)
        layout.addWidget(status_group)

        # Sky quality
        sky_group  = QGroupBox("Sky Quality")
        sky_layout = QVBoxLayout(sky_group)
        self.combo_sky = QComboBox()
        for name in SKY_PRESETS:
            self.combo_sky.addItem(name)
        self.combo_sky.currentTextChanged.connect(self._on_sky_changed)
        sky_layout.addWidget(self.combo_sky)
        self.lbl_sky_desc = QLabel("")
        self.lbl_sky_desc.setObjectName("info")
        self.lbl_sky_desc.setWordWrap(True)
        sky_layout.addWidget(self.lbl_sky_desc)
        layout.addWidget(sky_group)

        # Stacking method
        stack_group  = QGroupBox("Stacking Method")
        stack_layout = QVBoxLayout(stack_group)
        self.combo_stack = QComboBox()
        for idx, (name, cfg) in enumerate(STACKING_METHODS.items()):
            self.combo_stack.addItem(name)
            if "tooltip" in cfg:
                self.combo_stack.setItemData(idx, cfg["tooltip"], Qt.ItemDataRole.ToolTipRole)
        self.combo_stack.currentTextChanged.connect(self._on_stack_changed)
        stack_layout.addWidget(self.combo_stack)
        self.lbl_stack_desc = QLabel("")
        self.lbl_stack_desc.setObjectName("info")
        self.lbl_stack_desc.setWordWrap(True)
        stack_layout.addWidget(self.lbl_stack_desc)
        layout.addWidget(stack_group)

        # Stacking options
        opts_group  = QGroupBox("Stacking")
        opts_layout = QVBoxLayout(opts_group)

        hbox_bg2p = QHBoxLayout()
        self.chk_bg_extract = QCheckBox("Background Extraction")
        self.chk_bg_extract.setToolTip("Extract and subtract background before stacking")
        self.chk_two_pass = QCheckBox("2‑Pass Registration")
        self.chk_two_pass.setToolTip("Two‑pass registration with framing for optimal alignment")
        hbox_bg2p.addWidget(self.chk_bg_extract)
        hbox_bg2p.addWidget(self.chk_two_pass)
        opts_layout.addLayout(hbox_bg2p)

        feather_row = QHBoxLayout()
        self.chk_feather    = QCheckBox("Feathering")
        self.chk_feather.setToolTip("Blend image edges to reduce stacking artifacts (0–50 px)")
        self.feather_slider = QSpinBox()
        self.feather_slider.setRange(0, 50)
        self.feather_slider.setValue(0)
        self.feather_slider.setSuffix(" px")
        feather_row.addWidget(self.chk_feather)
        feather_row.addWidget(self.feather_slider)
        opts_layout.addLayout(feather_row)

        batch_row = QHBoxLayout()
        self.chk_batch = QCheckBox("Batch Processing")
        self.chk_batch.setToolTip("Split lights into batches (recommended for large sessions)")
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(10, 100)
        self.spin_batch_size.setValue(0)
        self.spin_batch_size.setSuffix(" images/chunk")
        batch_row.addWidget(self.chk_batch)
        batch_row.addWidget(self.spin_batch_size)
        opts_layout.addLayout(batch_row)
        layout.addWidget(opts_group)

        # Post‑stacking options
        post_group  = QGroupBox("Post‑Stacking")
        post_layout = QVBoxLayout(post_group)

        spcc_row = QHBoxLayout()
        self.spcc_cb = QCheckBox("SPCC")
        self.spcc_cb.setToolTip(
            "Calibrate colors using Gaia star catalog.\nProduces accurate, natural star colors."
        )
        self.spcc_filter_combo = QComboBox()
        self.spcc_filter_combo.addItems(["No Filter", "Dual Band Ha/Oiii", "City Light Pollution"])
        spcc_row.addWidget(self.spcc_cb)
        spcc_row.addWidget(self.spcc_filter_combo)
        post_layout.addLayout(spcc_row)

        auto_clean_row = QHBoxLayout()
        self.autostretch_cb = QCheckBox("Auto‑Stretch")
        self.autostretch_cb.setToolTip("Auto‑stretch image (linked).")
        self.chk_clean_temp = QCheckBox("Clean temporary files")
        self.chk_clean_temp.setToolTip("Delete process/ and masters/ folders after completion")
        auto_clean_row.addWidget(self.autostretch_cb)
        auto_clean_row.addWidget(self.chk_clean_temp)
        post_layout.addLayout(auto_clean_row)
        layout.addWidget(post_group)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    def _create_info_tab(self) -> QWidget:
        """Return the help / information tab."""
        widget = QWidget()
        outer  = QVBoxLayout(widget)
        outer.setSpacing(6)
        outer.setContentsMargins(6, 6, 6, 6)

        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: #2b2b2b; width: 10px; border-radius: 5px; }"
            "QScrollBar::handle:vertical { background: #555; border-radius: 5px; }"
        )

        container        = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(10)
        container_layout.setContentsMargins(4, 4, 4, 4)

        def make_section(title: str, html_body: str) -> QGroupBox:
            box    = QGroupBox(title)
            layout = QVBoxLayout(box)
            lbl    = QLabel(html_body)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #cccccc; font-size: 10pt;")
            lbl.setOpenExternalLinks(True)
            layout.addWidget(lbl)
            return box

        container_layout.addWidget(make_section(
            "Quick Start",
            """
            <p><b>No Setup Required.</b> Point the plugin at your Vespera observation
            folder and press <b>Start Processing</b>. The plugin auto-detects your folder
            layout and telescope model from the FITS headers.</p>
            <ul style="margin-left:14px;">
              <li><b>Folder detection</b> – recognises three layouts automatically:
                <ul style="margin-left:12px; margin-top:2px;">
                  <li>Organised: <code>darks/</code> &amp; <code>lights/</code> sub-folders.</li>
                  <li>Native Vespera: <code>01-images-initial/</code> + <code>*-dark.fits</code>.</li>
                  <li>Flat: all FITS in one folder – reorganised automatically.</li>
                </ul>
              </li>
              <li><b>Sky quality</b> – choose your Bortle rating; sigma-clipping thresholds
                  are set accordingly.</li>
              <li><b>Output</b> – a FITS image named <code>result_<i>exposure</i>s.fit</code>
                  (or <code>final_stacked_batch.fit</code> in batch mode) is written
                  to the working directory.</li>
            </ul>
            """
        ))

        container_layout.addWidget(make_section(
            "Stacking Options",
            """
            <table cellspacing="0" cellpadding="3" width="100%">
              <tr>
                <td width="34%" valign="top"><b>Bayer Drizzle<br>(Recommended)</b></td>
                <td valign="top">Gaussian kernel + area interpolation. Best choice for
                  typical Vespera sessions with 10–15° field rotation. Preserves smooth
                  PSFs and suppresses moiré at CFA boundaries.</td>
              </tr>
              <tr>
                <td valign="top"><b>Bayer Drizzle (Square)</b></td>
                <td valign="top">Original HST square kernel. Mathematically
                  flux-preserving – preferred for photometry. May show subtle grid
                  patterns when field rotation is large.</td>
              </tr>
              <tr>
                <td valign="top"><b>Bayer Drizzle (Nearest)</b></td>
                <td valign="top">Nearest-neighbour interpolation; zero interpolation
                  artifact at CFA boundaries. Try this if other drizzle modes show
                  a checkerboard pattern.</td>
              </tr>
              <tr>
                <td valign="top"><b>Standard Registration</b></td>
                <td valign="top">Classic debayer → register workflow (no drizzle).
                  Faster and lighter on RAM. Suitable for sessions under ~30 min with
                  less than 5° total rotation.</td>
              </tr>
              <tr>
                <td valign="top"><b>Drizzle 2× Upscale</b></td>
                <td valign="top">Doubles output resolution to 7072 × 7072 px. Requires
                  50+ well-dithered frames. Uses the square kernel (the only valid
                  choice when scale &gt; 1). Significantly increases processing time
                  and file size.</td>
              </tr>
              <tr><td colspan="2">&nbsp;</td></tr>
              <tr>
                <td valign="top"><b>Background Extraction</b></td>
                <td valign="top">Fits and subtracts a smooth RBF sky background
                  (60 samples, tolerance 1.0, smooth 0.5) from each calibrated frame
                  before registration. Reduces gradient artefacts on light-polluted or
                  uneven skies.</td>
              </tr>
              <tr>
                <td valign="top"><b>2-Pass Registration</b></td>
                <td valign="top">Runs a second alignment pass with
                  <code>-framing=max</code>, preserving the maximum common field of
                  view. Improves alignment when field rotation across the session is
                  significant. Not recommended with very small chunks (&lt; 20 frames).</td>
              </tr>
              <tr>
                <td valign="top"><b>Feathering</b></td>
                <td valign="top">Blends the edges of stacked sub-images with a soft
                  fall-off (0–50 px). Reduces hard seam artefacts in large mosaics
                  or sessions with strong field rotation. 10–20 px is a good starting
                  point; increase if seams remain visible.</td>
              </tr>
              <tr>
                <td valign="top"><b>Batch Processing</b></td>
                <td valign="top">Splits the light frames into smaller chunks, stacks
                  each independently, then combines the results. Reduces peak disk
                  usage significantly for large sessions. Minimum recommended chunk
                  size: <b>20 frames</b>. 2-Pass registration works best with larger
                  chunks.</td>
              </tr>
            </table>
            """
        ))

        container_layout.addWidget(make_section(
            "Post-Stacking Options",
            """
            <p>These steps run automatically on the final stacked image after
            calibration and registration are complete.</p>
            <table cellspacing="0" cellpadding="3" width="100%">
              <tr>
                <td width="34%" valign="top"><b>SPCC</b><br>
                  <i style="color:#888; font-size:9pt;">Spectrophotometric<br>Color Calibration</i>
                </td>
                <td valign="top">
                  Produces photometrically accurate, natural star colours by matching
                  measured star fluxes against the Gaia DR3 spectrophotometric catalogue.
                  <br><br>
                  <b>Requires plate-solving</b> – the plugin runs <code>platesolve</code>
                  automatically beforehand using the telescope focal length and pixel size
                  read from the FITS header. RA/DEC coordinates are looked up in SIMBAD
                  using the <code>OBJECT</code> keyword and written into the FITS header
                  before solving. If plate-solving fails, SPCC is skipped and a warning
                  is logged.
                  <br><br>
                  <b>Filter options</b>
                  <ul style="margin-left:12px; margin-top:3px;">
                    <li><b>No Filter</b> – broadband imaging under dark or mildly
                      light-polluted skies. Uses sensor spectral response only.</li>
                    <li><b>City Light Pollution (CLS)</b> – applies the Vaonis CLS
                      filter transmission curve; compensates for sodium / mercury
                      blocking.</li>
                    <li><b>Dual Band Ha/Oiii</b> – narrowband mode calibrated for
                      Hα 656.3 nm (BW 12 nm) in red and [O III] 500.7 nm (BW 12 nm)
                      in green; blue channel left unfiltered.</li>
                  </ul>
                </td>
              </tr>
              <tr><td colspan="2">&nbsp;</td></tr>
              <tr>
                <td valign="top"><b>Auto-Stretch</b></td>
                <td valign="top">
                  Applies Siril's linked auto-stretch (Midtone Transfer Function) so
                  the saved FITS is immediately viewable without a separate stretch step.
                  <br><br>
                  Parameters: shadows clip <code>−2.8σ</code>, target background
                  <code>0.25</code>. The stretch is applied <i>linked</i> (same curve
                  to all three channels), preserving the colour balance set by SPCC.
                  <br><span style="color:#ffaa44;">⚠ Stretch is non-reversible in the
                  saved file – disable if you want to keep the linear FITS for further
                  processing in Siril or PixInsight.</span>
                </td>
              </tr>
              <tr><td colspan="2">&nbsp;</td></tr>
              <tr>
                <td valign="top"><b>Clean temporary files</b></td>
                <td valign="top">
                  Deletes <code>process/</code>, <code>masters/</code>, and
                  <code>final_stack/</code> once processing completes successfully.
                  Reclaims several gigabytes of intermediate FITS data.
                  <br><span style="color:#ffaa44;">⚠ Leave this off until you are
                  confident the result is correct – deleted files cannot be recovered
                  without re-running the full pipeline.</span>
                </td>
              </tr>
            </table>
            """
        ))

        container_layout.addWidget(make_section(
            "Full Pipeline Order",
            """
            <ol style="margin-left:16px;">
              <li>Clean previous intermediate files <i>(if enabled)</i></li>
              <li>Build master dark (stack or copy single dark frame)</li>
              <li>Auto-detect telescope from FITS <code>NAXIS1/NAXIS2</code> dimensions</li>
              <li>Convert light frames to Siril sequence</li>
              <li>Calibrate – subtract dark, optionally debayer &amp; equalise CFA</li>
              <li>Background extraction <i>(if enabled)</i></li>
              <li>Register – 1-pass or 2-pass with framing</li>
              <li>Stack – sigma-clip rejection, additive scaling, FWHM weighting,
                optional feathering</li>
              <li><i>Batch mode only:</i> repeat steps 4–8 per chunk, then
                re-register and combine all chunk results</li>
              <li>Look up RA/DEC in SIMBAD and write to FITS header <i>(if SPCC enabled)</i></li>
              <li>Plate solve <i>(if SPCC enabled)</i></li>
              <li>SPCC colour calibration <i>(if plate-solve succeeded)</i></li>
              <li>Auto-stretch <i>(if enabled)</i></li>
              <li>Clean temporary files <i>(if enabled)</i></li>
            </ol>
            """
        ))

        container_layout.addWidget(make_section(
            "Known Limitations",
            """
            <ul style="margin-left:14px;">
              <li>Very small frame counts (&lt; 20) can cause registration or stacking
                to fail – increase the batch chunk size.</li>
              <li>2-Pass registration requires large chunks to compute a reliable
                reference frame; avoid it with chunks smaller than ~50 frames.</li>
              <li>Drizzle 2× Upscale ignores the 2-Pass and feathering flags for
                intermediate stacks; they are applied only during the final combine.</li>
              <li>SPCC requires an active internet connection to reach the Gaia /
                SIMBAD catalogues. Offline use will skip plate-solving and colour
                calibration.</li>
              <li>Auto-Stretch is irreversible in the written FITS file. Disable it
                if you intend to process the linear stack further.</li>
            </ul>
            """
        ))

        container_layout.addWidget(make_section(
            "Credits",
            """
            <p>Based on Siril's <code>OSC_Preprocessing_BayerDrizzle.ssf</code>.<br>
            Optimised for Vaonis Vespera II and Vespera Pro telescopes.<br>
            Developed for Siril. &copy; G. Trainar (2026) – MIT License.</p>
            """
        ))

        container_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)
        return widget

    # ------------------------------------------------------------------
    # Combo‑box callbacks
    # ------------------------------------------------------------------
    def _on_sky_changed(self, name: str) -> None:
        if name in SKY_PRESETS:
            self.lbl_sky_desc.setText(SKY_PRESETS[name]["description"])

    def _on_stack_changed(self, name: str) -> None:
        if name in STACKING_METHODS:
            self.lbl_stack_desc.setText(STACKING_METHODS[name]["description"])

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def _load_settings(self) -> None:
        get = self.qsettings.value
        self.combo_sky.setCurrentText(get("sky_quality",      "Bortle 3-4 (Rural)"))
        self.combo_stack.setCurrentText(get("stacking_method", "Bayer Drizzle (Recommended)"))
        self.chk_bg_extract.setChecked( get("bge",             False, type=bool))
        self.feather_slider.setValue(   get("feather_px",      0,     type=int))
        self.chk_feather.setChecked(    get("feather_enabled", False, type=bool))
        self.chk_two_pass.setChecked(   get("two_pass",        False, type=bool))
        self.chk_clean_temp.setChecked( get("clean_temp",      False, type=bool))
        self.chk_batch.setChecked(      get("batch_enabled",   False, type=bool))
        self.spin_batch_size.setValue(  get("batch_size",      100,   type=int))
        self.spcc_cb.setChecked(        get("spcc",            False, type=bool))
        self.autostretch_cb.setChecked( get("autostretch",     True,  type=bool))

    def _save_settings(self) -> None:
        set_ = self.qsettings.setValue
        set_("sky_quality",     self.combo_sky.currentText())
        set_("stacking_method", self.combo_stack.currentText())
        set_("bge",             self.chk_bg_extract.isChecked())
        set_("feather_px",      self.feather_slider.value())
        set_("feather_enabled", self.chk_feather.isChecked())
        set_("two_pass",        self.chk_two_pass.isChecked())
        set_("clean_temp",      self.chk_clean_temp.isChecked())
        set_("batch_enabled",   self.chk_batch.isChecked())
        set_("batch_size",      self.spin_batch_size.value())
        set_("spcc",            self.spcc_cb.isChecked())
        set_("autostretch",     self.autostretch_cb.isChecked())
        set_("spcc_filter",     self.spcc_filter_combo.currentText())

    # ------------------------------------------------------------------
    # Folder detection
    # ------------------------------------------------------------------
    def _browse_working_directory(self) -> None:
        """Open a directory chooser, update Siril's CWD, and refresh folder status."""
        selected = QFileDialog.getExistingDirectory(
            self, "Select Working Directory",
            self.workdir or os.path.expanduser("~"),
        )
        if selected:
            try:
                self.siril.cmd("cd", f'"{selected}"')
                self.workdir = selected
            except Exception as exc:
                self._log(f"Could not change Siril working dir: {exc}", LogColor.RED)
            self._check_folders()

    def _check_folders(self) -> None:
        """
        Auto‑detect the folder layout (organised, native Vespera, or flat)
        and update the status labels and Start button accordingly.
        """
        try:
            workdir = self.siril.get_siril_wd()
            self.workdir = workdir
            self.lbl_workdir.setText(f"Working directory: {workdir}")

            darks_dir  = os.path.join(workdir, "darks")
            lights_dir = os.path.join(workdir, "lights")

            num_darks_org  = count_fits_in(darks_dir)  if os.path.exists(darks_dir)  else 0
            num_lights_org = count_fits_in(lights_dir) if os.path.exists(lights_dir) else 0

            native = self._detect_native_structure(workdir)

            if num_darks_org > 0 and num_lights_org > 0:
                self.folder_structure = "organized"
                num_darks, num_lights = num_darks_org, num_lights_org
                self._set_structure_label("Using organized folders (darks/, lights/)")
            elif native:
                self.folder_structure = "native"
                num_darks  = native["num_darks"]
                num_lights = native["num_lights"]
                self._set_structure_label("Using Vespera native structure")
            elif self._organise_flat_directory(workdir):
                self.folder_structure = "organized"
                num_darks  = count_fits_in(os.path.join(workdir, "darks"))
                num_lights = count_fits_in(os.path.join(workdir, "lights"))
                self._set_structure_label("Automatically organised flat directory")
            else:
                self.folder_structure = None
                num_darks = num_lights = 0
                self._set_structure_label("No valid folder structure detected", error=True)

            self._update_count_label(self.lbl_darks,  "Darks",  num_darks)
            self._update_count_label(self.lbl_lights, "Lights", num_lights)
            self.btn_start.setEnabled(num_darks > 0 and num_lights > 0)

        except Exception as exc:
            self._log(f"Error: {exc}")
            self.btn_start.setEnabled(False)

    def _set_structure_label(self, text: str, error: bool = False) -> None:
        self.lbl_structure.setText(text)
        self.lbl_structure.setStyleSheet(
            "color: #ff8888;" if error else "color: #88aaff;"
        )

    def _update_count_label(self, label: QLabel, kind: str, count: int) -> None:
        if count > 0:
            label.setText(f"✓ {kind}: {count}")
            label.setStyleSheet("color: #88ff88;")
        else:
            label.setText(f"✗ {kind}: not found")
            label.setStyleSheet("color: #ff8888;")

    # ------------------------------------------------------------------
    def _detect_native_structure(self, workdir: str) -> Optional[Dict[str, Any]]:
        """Return metadata dict if the native Vespera layout is detected, else None."""
        dark_files = [
            p for ext in ("*-dark.fits", "*-dark.fit", "*-dark.FITS", "*-dark.FIT")
            for p in glob.glob(os.path.join(workdir, ext))
        ]
        images_initial = os.path.join(workdir, "01-images-initial")
        light_files = [
            f for ext in ("*.fits", "*.fit", "*.FITS", "*.FIT")
            for f in glob.glob(os.path.join(images_initial, ext))
            if "-dark" not in f.lower()
        ] if os.path.exists(images_initial) else []

        if dark_files and light_files:
            return {
                "dark_files":     dark_files,
                "light_files":    light_files,
                "num_darks":      len(dark_files),
                "num_lights":     len(light_files),
                "images_initial": images_initial,
            }
        return None

    # ------------------------------------------------------------------
    def _organise_flat_directory(self, workdir: str) -> bool:
        """
        Move FITS files from a flat directory into ``lights/`` and ``darks/``
        sub‑folders.  Returns True if reorganisation occurred.
        """
        fits_files = glob.glob(os.path.join(workdir, "*.fits"))
        if not fits_files:
            return False

        dark_patterns = ("master_dark.fits", "master-dark.fits", "*dark*.fits")
        dark_files  = [f for f in fits_files if any(
            glob.fnmatch.fnmatch(os.path.basename(f).lower(), p) for p in dark_patterns
        )]
        light_files = [f for f in fits_files if f not in dark_files]

        if not dark_files or not light_files:
            return False

        lights_dir = os.path.join(workdir, "lights")
        darks_dir  = os.path.join(workdir, "darks")
        os.makedirs(lights_dir, exist_ok=True)
        os.makedirs(darks_dir,  exist_ok=True)

        for src in light_files:
            shutil.move(src, os.path.join(lights_dir, os.path.basename(src)))
        for src in dark_files:
            shutil.move(src, os.path.join(darks_dir,  os.path.basename(src)))

        self._log(
            f"Organised flat directory: {len(light_files)} light(s) → lights/, "
            f"{len(dark_files)} dark(s) → darks/",
            LogColor.GREEN,
        )
        return True

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def _build_settings(self) -> Dict[str, Any]:
        """Collect all widget values into a single settings dict."""
        return {
            "sky_quality":     self.combo_sky.currentText(),
            "stacking_method": self.combo_stack.currentText(),
            "bge":             self.chk_bg_extract.isChecked(),
            "feather_px":      self.feather_slider.value(),
            "feather_enabled": self.chk_feather.isChecked(),
            "two_pass":        self.chk_two_pass.isChecked(),
            "clean_temp":      self.chk_clean_temp.isChecked(),
            "batch_enabled":   self.chk_batch.isChecked(),
            "batch_size":      self.spin_batch_size.value(),
            "spcc":            self.spcc_cb.isChecked(),
            "autostretch":     self.autostretch_cb.isChecked(),
            "focal_length_mm": 250.0,
            "pixel_size_um":   2.9,
            "spcc_sensor":     "Sony IMX585",
            "spcc_filter":     self.spcc_filter_combo.currentText(),
        }

    def _start_processing(self) -> None:
        """Validate, prepare, and launch the processing thread + disk monitor."""
        self._save_settings()
        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Processing...")
        self.log_area.clear()

        temp_dir = Path(self.siril.get_siril_wd()) / "Temp"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                self._log("Previous Temp directory cleaned", LogColor.BLUE)
            except Exception as exc:
                self._log(f"Could not delete Temp folder: {exc}", LogColor.RED)

        settings = self._build_settings()
        self.current_settings = settings

        try:
            workdir = self.siril.get_siril_wd()
            lights_dir = os.path.join(
                workdir,
                "lights" if self.folder_structure == "organized" else "01-images-initial",
            )
            num_lights = count_fits_in(lights_dir)

            if settings["batch_enabled"]:
                from math import ceil
                self.num_chunks = ceil(num_lights / settings["batch_size"])
            else:
                self.num_chunks = 1

            self.worker = ProcessingThread(self.siril, workdir, settings, self.folder_structure)
            self.worker.log_area = self.log_area
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.worker.log.connect(self._log)
            self.worker.start()

            logs_dir  = Path(workdir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            disk_log  = logs_dir / f"disk_usage_{timestamp}.log"
            with open(disk_log, "w", encoding="utf-8") as fh:
                fh.write(self._config_summary() + "\n")

            self.disk_thread = DiskUsageThread(disk_log, workdir=Path(workdir), interval_sec=5)
            self.disk_thread.start()

        except Exception as exc:
            self._log(f"Start error: {exc}", LogColor.RED)
            self.btn_start.setEnabled(True)

    # ------------------------------------------------------------------
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        self.status.setText(message)
        self.status.setStyleSheet("color: #ffcc00;")
        self.app.processEvents()

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_start.setEnabled(True)

        if hasattr(self, "disk_thread"):
            self.disk_thread.stop()
            self.disk_thread.wait()

        if hasattr(self, "worker"):
            self._write_console_log()

        if success:
            self.status.setText(f"✓ {message}")
            self.status.setStyleSheet("color: #88ff88;")
            try:
                self.siril.log("Stacking Complete!", color=LogColor.GREEN)
            except Exception:
                pass
        else:
            self.status.setText(f"✗ {message}")
            self.status.setStyleSheet("color: #ff8888;")
            self._log(f"FAILED: {message}", LogColor.RED)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _config_summary(self) -> str:
        s = self.current_settings
        return (
            f"Stacking method: {s.get('stacking_method', 'Unknown')}, "
            f"Feathering: {'Yes' if s.get('feather_enabled') else 'No'}, "
            f"2‑Pass: {'Yes' if s.get('two_pass') else 'No'}, "
            f"Batch: {'Yes' if s.get('batch_enabled') else 'No'}, "
            f"Chunks: {getattr(self, 'num_chunks', 1)}"
        )

    def _write_console_log(self) -> None:
        """Persist the worker's accumulated console messages to a timestamped log file."""
        if not hasattr(self, "worker"):
            return
        logs_dir = Path(self.workdir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"siril_console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with open(log_file, "w", encoding="utf-8") as fh:
                fh.write(self._config_summary() + "\n")
                fh.writelines(msg + "\n" for msg in self.worker.console_messages)
        except Exception as exc:
            self._log(f"Failed to write console log: {exc}", LogColor.RED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_global_gui: Optional[VesperaProGUI] = None


def main() -> None:
    """Launch the Vespera Preprocessing GUI."""
    global _global_gui

    try:
        app = QApplication.instance() or QApplication(sys.argv)

        siril = s.SirilInterface()
        try:
            siril.connect()
        except Exception as exc:
            QMessageBox.critical(None, "Connection Error", f"Could not connect to Siril.\n{exc}")
            return

        gui = VesperaProGUI(siril, app)
        _global_gui = gui

        def _crash_handler(exc_type, exc_value, tb):
            """Flush logs gracefully if the interpreter crashes."""
            if _global_gui:
                disk = getattr(_global_gui, "disk_thread", None)
                if disk:
                    try:
                        disk.stop()
                        disk.wait()
                    except Exception as exc:
                        _global_gui._log(f"Error stopping disk thread: {exc}", LogColor.RED)
                if hasattr(_global_gui, "worker"):
                    _global_gui._write_console_log()
            sys.__excepthook__(exc_type, exc_value, tb)

        sys.excepthook = _crash_handler
        gui.show()
        app.exec()

    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
