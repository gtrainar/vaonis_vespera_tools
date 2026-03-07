##############################################
# Vespera — Preprocessing
# Automated Stacking for Alt‑Az Mounts
##############################################
# (c) 2026 G. Trainar - MIT License
# Vespera Preprocessing
# Version 1.3.1
#
# Credits
# ----------------
#   • Based on Siril's OSC_Preprocessing_BayerDrizzle.ssf
#   • Optimized for Vaonis Vespera II and Pro telescopes
#   • Handles single dark frame capture (Expert Mode)
##############################################

import sys
import os
import glob
import shutil
import time
import re
import json
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

s.ensure_installed("PyQt6")

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog,
    QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget, QFrame, QSizePolicy,
)
# ---------------------------------------------------------------------------
# Version & Changelog
# ---------------------------------------------------------------------------
VERSION = "1.3.1"

CHANGELOG = """
Version 1.3.0 (2026-03)
- Memory management for large datasets
- Local plate solving if available

Version 1.3.0 (2026-03)
- Redesigned GUI: pipeline-staged layout
- Added "RICE 16 Compression"

Version 1.2.0 (2026-02)
- Post-Stacking Options (SPCC, Autostretch)
- GUI update
- Code refactoring

Version 1.1.0 (2026-02)
- Batch Processing for disk optimization
- File Dialog for Working Directory

Version 1.0.0 (2026‑01)
- ProcessingProgress constants for standardized progress tracking
- Implemented feathering option (0‑50px) to reduce stacking artifacts
- Added two‑pass registration with framing for improved field rotation handling
- Enhanced logging system with color‑coded messages (red/green/blue/salmon)
- Improved error handling and validation throughout the processing pipeline
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class ProcessingProgress:
    """Standardised progress percentages for each pipeline stage."""
    CLEANUP          = 5
    DARK_PROCESSING  = 10
    LIGHT_CONVERSION = 20
    CALIBRATION      = 25
    BACKGROUND_EXTR  = 30
    GET_COORDINATES  = 35
    PLATESOLVING     = 40
    REGISTRATION     = 45
    STACKING         = 55
    FINALIZATION     = 88
    TEMP_CLEANUP     = 95
    COMPLETE         = 100


# Sky quality presets keyed by Bortle description
SKY_PRESETS: Dict[str, Dict[str, Any]] = {
    "Bortle 1-3 (Dark Skies)": {
        "description": "Remote dark sites, minimal light pollution",
        "sigma_low": 2.8,
        "sigma_high": 3.2,
    },
    "Bortle 4-5 (Rural)": {
        "description": "Rural areas, some light domes on horizon",
        "sigma_low": 2.2,
        "sigma_high": 2.7,
    },
    "Bortle 6-7 (Suburban)": {
        "description": "Suburban skies, noticeable light pollution",
        "sigma_low": 1.8,
        "sigma_high": 2.2,
    },
    "Bortle 8-9 (Urban)": {
        "description": "City skies, heavy light pollution",
        "sigma_low": 1.5,
        "sigma_high": 2.0,
    },
}

# Stacking methods with technical metadata
STACKING_METHODS: Dict[str, Dict[str, Any]] = {
    "Bayer Drizzle (Recommended)": {
        "description": "Gaussian kernel + area interp — best for 10–15° field rotation",
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
        "description": "Square kernel — flux-preserving, preferred for photometry",
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
        "description": "Nearest-neighbour interp — eliminates moiré at CFA boundaries",
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
        "description": "No drizzle — fast, good for <30 min sessions with <5° rotation",
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
        "description": "2× output resolution — requires 50+ well-dithered frames",
        "tooltip": (
            "Upscales to 2x resolution using drizzle algorithm.\n\n"
            "• Requires 50+ frames with good sub‑pixel dithering\n"
            "• Output will be 2× the native sensor resolution\n"
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
# Dark stylesheet — refined for pipeline-staged layout
# ---------------------------------------------------------------------------
DARK_STYLESHEET = """
QDialog { background-color: #1e1e2e; color: #e0e0e0; }
QTabWidget::pane { border: 1px solid #383850; background-color: #1e1e2e; }
QTabBar::tab {
    background-color: #2a2a3e; color: #8888aa;
    padding: 7px 18px; border: 1px solid #383850;
    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #1e1e2e; color: #ffffff; border-bottom: 1px solid #1e1e2e; }
QTabBar::tab:hover    { background-color: #32324a; color: #ccccee; }

QGroupBox {
    border: 1px solid #383850; margin-top: 14px;
    border-radius: 5px; padding-top: 6px; padding-bottom: 0px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 6px;
    color: #6688cc; font-size: 9pt; font-weight: bold; letter-spacing: 0.5px;
}
QGroupBox#stage {
    border-left: 2px solid #4466aa;
}

QLabel             { color: #b0b0cc; font-size: 12pt; }
QLabel#hint        { color: #5566aa; font-size: 10pt; font-family: 'Menlo', 'Monaco', 'Courier New'; }
QLabel#sigma       { color: #44aa88; font-size: 10pt; font-family: 'Menlo', 'Monaco', 'Courier New'; }
QLabel#status      { color: #ffcc44; font-size: 11pt; }

QComboBox {
    background-color: #26263a; color: #ddddee; border: 1px solid #444466;
    border-radius: 4px; padding: 4px 10px;
}
QComboBox:hover { border-color: #6688cc; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #8888aa;
}
QComboBox QAbstractItemView {
    background-color: #26263a; color: #ddddee;
    selection-background-color: #334488; border: 1px solid #444466;
}

QCheckBox { color: #b0b0cc; spacing: 6px; font-size: 12pt; }
QCheckBox::indicator {
    width: 15px; height: 15px; border: 1px solid #555577;
    background: #26263a; border-radius: 3px;
}
QCheckBox::indicator:checked  { background-color: #334488; border: 1px solid #6688cc; }
QCheckBox::indicator:hover    { border-color: #6688cc; }

QSpinBox {
    background-color: #26263a; color: #ddddee; border: 1px solid #444466;
    border-radius: 4px; padding: 4px 5px; min-height: 15px;
}
QSpinBox:hover { border-color: #6688cc; }

QProgressBar {
    border: 1px solid #383850; border-radius: 4px; background-color: #26263a;
    text-align: center; color: #aaaacc; min-height: 18px; font-size: 9pt;
}
QProgressBar::chunk { background-color: #334488; border-radius: 3px; }

QPushButton {
    background-color: #2e2e44; color: #ccccdd; border: 1px solid #444466;
    border-radius: 4px; padding: 6px 18px; font-size: 12pt; min-width: 90px;
}
QPushButton:hover    { background-color: #383858; border-color: #6688cc; }
QPushButton:pressed  { background-color: #222236; }
QPushButton:disabled { background-color: #1e1e2e; color: #444455; border-color: #2a2a3e; }

QPushButton#start          { background-color: #1e3a6a; border: 1px solid #2a5090; color: #aaccff; font-weight: bold; }
QPushButton#start:hover    { background-color: #264888; border-color: #4477cc; }
QPushButton#start:disabled { background-color: #151525; color: #333355; border-color: #222235; }

QPushButton#browse {
    background-color: transparent; color: #aaccff; border: 1px solid #333355;
    padding: 3px 10px; font-size: 11pt; min-width: 60px;
}
QPushButton#browse:hover { border-color: #6688cc; color: #88aaff; }

QTextEdit {
    background-color: #14141e; color: #8888aa; border: 1px solid #2a2a3e;
    border-radius: 4px; font-family: 'Menlo', 'Monaco', 'Courier New';
    font-size: 9pt; padding: 4px;
}

QFrame#hline { background-color: #2a2a3e; min-height: 1px; max-height: 1px; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_fits_in(folder: str) -> int:
    """Return the number of FITS files in *folder* (all common extensions)."""
    return sum(
        len(glob.glob(os.path.join(folder, ext)))
        for ext in ("*.fit", "*.fits", "*.FIT", "*.FITS", "*.fz")
    )

def append_colored_text(text_edit: QTextEdit, msg: str, color: Optional[LogColor]) -> None:
    """Append *msg* to *text_edit* using the colour that corresponds to *color*."""
    cursor = text_edit.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    text_edit.setTextCursor(cursor)
    qt_color = _LOG_COLOR_MAP.get(color, Qt.GlobalColor.lightGray)
    text_edit.setTextColor(qt_color)
    text_edit.append(msg)
    text_edit.setTextColor(Qt.GlobalColor.lightGray)

def make_hline() -> QFrame:
    """Return a thin horizontal separator line."""
    line = QFrame()
    line.setObjectName("hline")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


# ---------------------------------------------------------------------------
# Processing Thread
# ---------------------------------------------------------------------------
class ProcessingThread(QThread):
    """Background thread that runs the full Siril preprocessing pipeline."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log      = pyqtSignal(str)

    def __init__(self, siril: Any, workdir: str, settings: Dict[str, Any],
                 folder_structure: str, gaia_available: bool = False) -> None:
        super().__init__()
        self.siril                         = siril
        self.workdir                       = workdir
        self.settings                      = settings
        self.folder_structure              = folder_structure
        self.local_catalog_gaia_available  = gaia_available
        self.log_area: Optional[QTextEdit] = None
        self.console_messages: List[str]   = []
        self.light_seq_name  = "light"
        self.final_filename  = ""

    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        if self.log_area:
            append_colored_text(self.log_area, msg, color)
        try:
            self.siril.log(msg, color=color) if color else self.siril.log(msg)
        except Exception as exc:
            self.console_messages.append(f"Logging error: {exc}")
        self.console_messages.append(msg)

    def _run(self, *cmd: str) -> bool:
        try:
            self.siril.cmd(*cmd)
            return True
        except (RuntimeError, Exception) as exc:
            self._log(f"Error in '{' '.join(cmd)}': {exc}", LogColor.RED)
            return False
    
    @property
    def _process_dir(self) -> str:
        return os.path.normpath(os.path.join(self.workdir, "process"))

    @property
    def _masters_dir(self) -> str:
        return os.path.normpath(os.path.join(self.workdir, "masters"))

    @property
    def _lights_dir(self) -> str:
        if getattr(self, 'folder_structure', None) == "native":
            return os.path.normpath(os.path.join(self.workdir, "01-images-initial"))
        return os.path.normpath(os.path.join(self.workdir, "lights"))

    def _create_final_stack_dir(self) -> Path:
        final_dir = Path(self.workdir, "final_stack")
        try:
            final_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Could not create final stack dir {final_dir}: {exc}")
        return final_dir

    def _move_tiff_to_reference(self, lights_dir: str) -> int:
        tiff_files = [
            f for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
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

    def _cleanup_folder(self, folder) -> int:
        folder = str(folder)
        if not os.path.exists(folder):
            return 0
        count = 0
        cleanup_patterns = ("*.fit", "*.fits", "*.FIT", "*.FITS", "*.fz", "*.seq", "*conversion.txt")
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

    def _cleanup_sequence_prefix(self, folder: str, prefix: str) -> None:
        """Remove intermediate sequence files."""
        for pattern in (f"{prefix}_*.fits", f"{prefix}_*.fit",
                        f"{prefix}_*.FITS", f"{prefix}_*.FIT", f"{prefix}.seq"):
            for filepath in glob.glob(os.path.join(folder, pattern)):
                try:
                    os.remove(filepath)
                except OSError as exc:
                    self._log(f"Warning: could not remove {filepath}: {exc}", LogColor.SALMON)

    def _find_best_frame(self, seq_name: str, process_dir: str) -> Optional[str]:
        """Return the path of the frame with the most detected stars."""
        self._log("Running SEQFINDSTAR to identify best frame", LogColor.BLUE)
        try:
            self.siril.cmd("seqfindstar", seq_name, "-maxstars=100")
        except Exception as exc:
            self._log(f"SEQFINDSTAR failed: {exc} — skipping best-frame selection",
                      LogColor.SALMON)
            return None

        lst_files = sorted(glob.glob(os.path.join(process_dir, "cache", f"{seq_name}_*.lst")))
        if not lst_files:
            self._log("No .lst files found after SEQFINDSTAR — skipping best-frame selection",
                      LogColor.SALMON)
            return None

        best_lst:   Optional[str] = None
        best_count: int           = -1

        for lst_path in lst_files:
            try:
                with open(lst_path, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                star_count = sum(
                    1 for ln in lines
                    if ln.strip() and not ln.strip().startswith("#")
                )
                if star_count > best_count:
                    best_count = star_count
                    best_lst   = lst_path
            except OSError as exc:
                self._log(f"Could not read {lst_path}: {exc}", LogColor.SALMON)

        for lst_path in lst_files:
            try:
                os.remove(lst_path)
            except OSError as exc:
                self._log(f"Warning: could not remove {lst_path}: {exc}", LogColor.SALMON)
        self._log(f"Removed {len(lst_files)} .lst file(s) from cache/", LogColor.BLUE)

        if best_lst is None or best_count <= 0:
            self._log("All .lst files empty — skipping best-frame selection", LogColor.SALMON)
            return None

        lst_stem  = os.path.splitext(os.path.basename(best_lst))[0]
        fits_path = os.path.join(process_dir, f"{lst_stem}.fits")
        if not os.path.exists(fits_path):
            fits_path = os.path.join(process_dir, f"{lst_stem}.fit")
        if not os.path.exists(fits_path):
            self._log(f"Best frame FITS not found for {lst_stem} — skipping best-frame selection",
                      LogColor.SALMON)
            return None

        self._log(f"Best frame: {lst_stem}.fits  ({best_count} stars detected)", LogColor.BLUE)
        return fits_path

    def _blind_solve(self, file_path: str):
        """Load file_path into Siril, run blind plate-solve, return (ra_deg, dec_deg)."""
        try:
            self.siril.cmd("load", f'"{file_path}"')
            self.siril.cmd("platesolve", "-localasnet", "-blindpos")
            self.siril.cmd("close")
        except Exception as e:
            self.siril.log(f"Could not plate solve the image: {e}", LogColor.SALMON)
            return None
        try:
            header = self.siril.get_image_fits_header(return_as="dict")
            if isinstance(header, dict):
                ra  = header.get("RA",  0)
                dec = header.get("DEC", 0)
                return float(ra), float(dec)
        except Exception as e:
            self.siril.log(f"Could not read RA/DEC from header: {e}", LogColor.SALMON)
        return None
    
    def _get_coordinates(self, lights_dir: str,
                         seq_name:    Optional[str] = None,
                         process_dir: Optional[str] = None) -> Optional[tuple]:
        """Return (RA, DEC) in degrees via blind solve → SEQFINDSTAR → SIMBAD cascade."""
        light_files = sorted(
            f for f in os.listdir(lights_dir)
            if f.lower().endswith((".fits", ".fit"))
        )

        if self.local_catalog_gaia_available:
            # Step 1 — blind solve on first raw light
            if light_files:
                first_light = os.path.join(lights_dir, light_files[0])
                self.siril.log(f"Blind solving first raw light: {light_files[0]}", LogColor.BLUE)
                coords = self._blind_solve(first_light)
                if coords:
                    self.siril.log(f"Blind solve succeeded: RA={coords[0]:.6f}  DEC={coords[1]:.6f}",
                                   LogColor.BLUE)
                    return coords
                self.siril.log("Blind solve failed — trying best frame", LogColor.SALMON)
            else:
                self.siril.log("No light files found — skipping blind solve", LogColor.SALMON)

            # Step 2 — blind solve on best frame via SEQFINDSTAR
            if seq_name and process_dir:
                try:
                    self.siril.cmd("cd", f'"{process_dir}"')
                except Exception:
                    pass
                best_frame = self._find_best_frame(seq_name, process_dir)
                if best_frame:
                    self.siril.log(f"Blind solving best frame: {os.path.basename(best_frame)}",
                                   LogColor.BLUE)
                    coords = self._blind_solve(best_frame)
                    if coords:
                        self.siril.log(f"Blind solve succeeded: RA={coords[0]:.6f}  DEC={coords[1]:.6f}",
                                       LogColor.BLUE)
                        return coords
                    self.siril.log("Blind solve failed on best frame — trying SIMBAD", LogColor.SALMON)
                else:
                    self.siril.log("Best-frame selection unavailable — trying SIMBAD", LogColor.SALMON)
        else:
            self.siril.log("Local Gaia catalogue not available — skipping blind solve, trying SIMBAD",
                           LogColor.SALMON)

        # Step 3 — SIMBAD via OBJECT header
        if not light_files:
            self.siril.log("Could not determine coordinates — plate solving will be skipped",
                           LogColor.SALMON)
            return None

        try:
            self.siril.cmd("load", f'"{os.path.join(lights_dir, light_files[0])}"')
        except Exception as exc:
            self._log(f"Could not load image for SIMBAD lookup: {exc}", LogColor.SALMON)
            return None
        try:
            header_dict = self.siril.get_image_fits_header(return_as='dict')
            obj_name = None
            if isinstance(header_dict, dict):
                obj_name = str(header_dict.get("OBJECT", "")).strip() or None
            self.siril.cmd("close")
            self.siril.cmd("cd", ".")
        except Exception as e:
            self._log(f"Could not read OBJECT header: {e}", LogColor.SALMON)
            return None

        if not obj_name:
            self._log("No OBJECT header — cannot query SIMBAD", LogColor.SALMON)
            return None

        try:
            base_url = "https://simbad.cds.unistra.fr/simbad/sim-id"
            url = f"{base_url}?{urllib.parse.urlencode({'output.format': 'ASCII', 'Ident': obj_name})}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode("utf-8")

            for line in data.splitlines():
                if line.startswith("Coordinates(ICRS,ep=J2000,eq=2000):"):
                    _, coord_part = line.split(":", 1)
                    m = re.search(
                        r"(\d+)\s+(\d+)\s+([\d.]+)\s+([+-]?\d+)\s+(\d+)\s+([\d.]+)",
                        coord_part,
                    )
                    if not m:
                        break
                    ra_h, ra_m, ra_s, dec_d, dec_m, dec_s = m.groups()
                    ra_deg  = 15.0 * (float(ra_h) + float(ra_m) / 60.0 + float(ra_s) / 3600.0)
                    sign    = -1 if dec_d.strip().startswith("-") else 1
                    dec_deg = sign * (abs(float(dec_d)) + float(dec_m) / 60.0 + float(dec_s) / 3600.0)
                    self.siril.log(
                        f"SIMBAD succeeded for '{obj_name}': RA={ra_deg:.6f}  DEC={dec_deg:.6f}",
                        LogColor.BLUE,
                    )
                    return ra_deg, dec_deg

            self.siril.log(f"SIMBAD returned no coordinates for '{obj_name}'", LogColor.SALMON)
        except Exception as e:
            self.siril.log(f"SIMBAD query error: {e}", LogColor.SALMON)

        self.siril.log("Could not determine coordinates — plate solving will be skipped",
                       LogColor.SALMON)
        return None


    def _set_telescope_from_fits(self) -> None:
        if getattr(self, 'folder_structure', None) == "native":
            lights_dir = os.path.join(self.workdir, "01-images-initial")
        else:
            lights_dir = os.path.join(self.workdir, "lights")

        if not os.path.exists(lights_dir):
            self.siril.log(f"Lights dir not found for telescope detection: {lights_dir}", LogColor.SALMON)
            return
        fits_files = [
            f for f in os.listdir(lights_dir)
            if f.lower().endswith((".fits", ".fit", ".fits.fz", ".fit.fz"))
        ]
        if not fits_files:
            return
        first_file = os.path.join(lights_dir, fits_files[0])
        try:
            self.siril.cmd("load", f'"{first_file}"')
        except Exception as exc:
            self.siril.log(f"Could not load FITS for telescope detection: {exc}", LogColor.SALMON)
            return
        try:
            header_dict = self.siril.get_image_fits_header(return_as='dict')
            naxis1 = 0
            naxis2 = 0
            if isinstance(header_dict, dict):
                naxis1 = header_dict.get("NAXIS1", 0)
                naxis2 = header_dict.get("NAXIS2", 0)
        except Exception as exc:
            self.siril.log(f"Error reading telescope from FITS: {exc}", LogColor.SALMON)
            return
        finally:
            try:
                self.siril.cmd("close")
            except Exception:
                pass
        if naxis1 == 3536 and naxis2 == 3536:
            model = "Vespera Pro"
        elif naxis1 == 3840 and naxis2 == 2160:
            model = "Vespera II"
        else:
            self.siril.log("Couldn't find telescope info, using defaults", LogColor.BLUE)
            return
        self.settings.update(TELESCOPES[model])
        self._log(f"Set telescope to {model} from FITS header", LogColor.BLUE)

    def _process(self) -> None:
        _start_time = time.time()
        try:
            self.siril.cmd("close")
        except Exception:
            pass

        sky      = SKY_PRESETS[self.settings["sky_quality"]]
        method   = STACKING_METHODS[self.settings["stacking_method"]]
        sigma_lo = sky["sigma_low"]
        sigma_hi = sky["sigma_high"]

        # Validate input folders
        if self.folder_structure == "native":
            dark_file  = os.path.join(self.workdir, "img-0001-dark.fits")
            lights_dir = os.path.join(self.workdir, "01-images-initial")
        else:
            darks_dir  = os.path.join(self.workdir, "darks")
            lights_dir = os.path.join(self.workdir, "lights")

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

        # Create working directories
        process_dir = self._process_dir
        masters_dir = self._masters_dir
        for d in (process_dir, masters_dir):
            os.makedirs(d, exist_ok=True)

        # Move reference images
        moved = self._move_tiff_to_reference(lights_dir)
        if moved:
            self._log(f"Moved {moved} TIFF reference image(s) to 'reference/'", LogColor.SALMON)

        # Count frames
        if self.folder_structure == "native":
            num_darks = 1
            num_lights = len([
                f for f in (
                    glob.glob(os.path.join(lights_dir, "*.fits")) +
                    glob.glob(os.path.join(lights_dir, "*.fit"))
                )
                if "-dark" not in f.lower()
            ])
        else:
            num_darks  = count_fits_in(darks_dir)
            num_lights = count_fits_in(lights_dir)

        for label, count in (("dark", num_darks), ("light", num_lights)):
            if count == 0:
                self.finished.emit(False, f"No {label} frames found")
                return

        self._log(f"Sky Quality: {self.settings['sky_quality']}",    LogColor.BLUE)
        self._log(f"Stacking:    {self.settings['stacking_method']}", LogColor.BLUE)
        self._log(f"Structure:   {self.folder_structure}",            LogColor.BLUE)
        self._log(f"Found {num_darks} dark(s), {num_lights} light(s)", LogColor.BLUE)

        # Cleanup
        self.progress.emit(ProcessingProgress.CLEANUP, "Cleaning previous files")
        deleted = self._cleanup_folder(process_dir) + self._cleanup_folder(masters_dir)
        self._log(f"Cleaned {deleted} temp files", LogColor.BLUE)

        # Process darks
        self.progress.emit(ProcessingProgress.DARK_PROCESSING, "Processing darks")
        if self.folder_structure == "native":
            self._log("Single dark → using directly as master", LogColor.BLUE)
            if not self._run("load", f'"{dark_file}"'):
                self.finished.emit(False, "Failed to load dark file")
                return
            if not self._run("save", "masters/dark_stacked"):
                self.finished.emit(False, "Failed to save master dark")
                return
        else:
            if not self._run("cd", "darks"):
                self.finished.emit(False, "Failed to enter darks directory")
                return
            if not self._run("convert", "dark", "-out=../masters"):
                self.finished.emit(False, "Failed to convert darks")
                return
            if not self._run("cd", "../masters"):
                self.finished.emit(False, "Failed to enter masters directory")
                return

            if num_darks == 1:
                self._log("Single dark → using directly as master", LogColor.BLUE)
                if not self._run("load", "dark_00001"):
                    self.finished.emit(False, "Failed to load dark frame")
                    return
                if not self._run("save", "dark_stacked"):
                    self.finished.emit(False, "Failed to save stacked dark")
                    return
            else:
                self._log(f"Stacking {num_darks} darks", LogColor.BLUE)
                if not self._run("stack", "dark", "rej", str(sigma_lo), str(sigma_hi),
                                 "-nonorm", "-out=dark_stacked"):
                    self.finished.emit(False, "Failed to stack darks")
                    return

        # Convert lights
        self.progress.emit(ProcessingProgress.LIGHT_CONVERSION, "Converting lights")
        if not self._run("cd", "01-images-initial" if self.folder_structure == "native" else "../lights"):
            self.finished.emit(False, "Failed to enter lights directory")
            return
        if not self._run("convert", "light", "-out=../process"):
            self.finished.emit(False, "Failed to convert lights")
            return
        if not self._run("cd", "../process"):
            self.finished.emit(False, "Failed to enter process directory")
            return
        self.light_seq_name = "light"

        # Telescope detection
        self._set_telescope_from_fits()

        # Batch or standard processing
        if self.settings.get("batch_enabled"):
            self._process_batch_sessions(method, sigma_lo, sigma_hi)
        else:
            self._process_standard(method, sigma_lo, sigma_hi)

        # Temp cleanup
        if self.settings.get("clean_temp"):
            self.progress.emit(ProcessingProgress.TEMP_CLEANUP, "Cleaning up")
            deleted = self._cleanup_folder(process_dir) + self._cleanup_folder(masters_dir)
            self._log(f"Cleaned {deleted} temp files", LogColor.BLUE)

        # Post-stacking
        self._run_poststacking()

        # Summary
        self._log_summary(time.time() - _start_time)
        self.progress.emit(ProcessingProgress.COMPLETE, "Complete!")
        self.finished.emit(True, "Processing complete!")

    def _process_standard(
        self,
        method: Dict[str, Any],
        sigma_low: float,
        sigma_high: float,
        chunk_idx: Optional[int] = None,
        total_chunks: Optional[int] = None,
        active_process_dir: Optional[str] = None,
    ) -> None:
        """Calibrate → (optional BGE) → Plate-solve → Register → Stack → Finalize.

        active_process_dir: actual on-disk process/ path for the current context
            (differs from self._process_dir in batch mode). Defaults to self._process_dir.
        """
        self.siril.cmd("close")
        _apd = active_process_dir if active_process_dir is not None else self._process_dir
        if active_process_dir is not None:
            self._run("cd", f'"{active_process_dir}"')

        def _emit(stage_pct: float, msg: str) -> None:
            """Emit progress, optionally scaled to the current chunk's slice of the bar."""
            if chunk_idx is not None and total_chunks:
                chunk_span = (ProcessingProgress.FINALIZATION - ProcessingProgress.CALIBRATION) / total_chunks
                start      = ProcessingProgress.CALIBRATION + (chunk_idx - 1) * chunk_span
                percent    = int(start + (stage_pct / ProcessingProgress.FINALIZATION) * chunk_span)
            else:
                percent = int(stage_pct)
            self.progress.emit(percent, msg)

        def _chunk_label(base: str) -> str:
            return f"{base} chunk {chunk_idx} of {total_chunks}" if chunk_idx else base

        # -- Drizzle safeguards -------------------------------------------
        if method.get("use_drizzle"):
            warnings = []
            if self.settings.get("stacking_method") == "Drizzle 2x Upscale":
                if self.settings.get("two_pass"):
                    self.settings["two_pass"] = False
                    warnings.append("2-Pass Registration")
            if self.settings.get("bge"):
                self.settings["bge"] = False
                warnings.append("BGE")
            if warnings:
                self._log(
                    f"Drizzle: {' and '.join(warnings)} disabled "
                    f"(incompatible — drizzle requires CFA input).",
                    LogColor.SALMON,
                )

        # -- RICE 16 Compression (optional) --------------------------------
        if self.settings.get("use_compression"):
            if self._run("setcompress", "1", "-type=rice", "16"):
                self._log("Rice 16 compression enabled", LogColor.BLUE)
            else:
                self._log("setcompress failed — continuing without compression", LogColor.SALMON)
                self.settings["use_compression"] = False

        # -- Calibration ---------------------------------------------------
        _emit(ProcessingProgress.CALIBRATION, _chunk_label("Calibrating"))
        try:
            self._calibrate(self.light_seq_name, method)
        except Exception as exc:
            self._log(f"Calibration failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)
            return

        reg_input = f"pp_{self.light_seq_name}"
        stack_seq = f"r_pp_{self.light_seq_name}"

        # -- Background extraction (optional, pre-registration) ------------
        if self.settings.get("bge"):
            _emit(ProcessingProgress.BACKGROUND_EXTR, _chunk_label("Background Extraction"))
            try:
                self._run_background_extraction(seq_name=reg_input)
                self._cleanup_sequence_prefix(_apd, f"pp_{self.light_seq_name}")
                self._log(f"Flushed pp_{self.light_seq_name} sequences after BGE", LogColor.BLUE)
                reg_input = f"bkg_pp_{self.light_seq_name}"
                stack_seq = f"r_bkg_pp_{self.light_seq_name}"
            except Exception as exc:
                self._log(
                    f"Background extraction failed"
                    f"{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc} — stacking without BGE",
                    LogColor.SALMON,
                )

        # -- Coordinate resolution + Plate solving -------------------------
        _emit(ProcessingProgress.GET_COORDINATES, _chunk_label("Resolving coordinates"))
        plate_solve_ok = False

        if self.local_catalog_gaia_available:
            try:
                self.simbad_coordinates = self._get_coordinates(
                    lights_dir  = self._lights_dir,
                    seq_name    = reg_input,
                    process_dir = _apd,
                )
                _emit(ProcessingProgress.PLATESOLVING, _chunk_label("Plate solving"))
                plate_solve_ok = self._seq_plate_solve(reg_input)
            except Exception as exc:
                self._log(f"Plate solving error: {exc}", LogColor.SALMON)
        else:
            self._log("Local Gaia catalogue not available — using regular registration.", LogColor.SALMON)

        # -- Registration --------------------------------------------------
        _emit(ProcessingProgress.REGISTRATION, _chunk_label("Registering"))
        try:
            if plate_solve_ok:
                self._seq_apply_reg(reg_input, method)
            else:
                self._register_regular(reg_input, method)
        except Exception as exc:
            self._log(f"Registration failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)
            return

        # -- Pre-stacking sequence flush ------------------------------------
        use_drizzle = method.get("use_drizzle", False)
        if use_drizzle:
            self._cleanup_sequence_prefix(_apd, self.light_seq_name)
            self._cleanup_sequence_prefix(_apd, f"pp_{self.light_seq_name}")
            self._log(
                f"Flushed {self.light_seq_name} and pp_{self.light_seq_name} sequences before drizzle stack",
                LogColor.BLUE,
            )
        else:
            self._cleanup_sequence_prefix(_apd, self.light_seq_name)
            if self.settings.get("bge"):
                self._cleanup_sequence_prefix(_apd, f"bkg_pp_{self.light_seq_name}")
            else:
                self._cleanup_sequence_prefix(_apd, f"pp_{self.light_seq_name}")
            self._log("Flushed pre-registration sequences before stacking", LogColor.BLUE)

        # -- Unset RICE 16 before stacking ---------------------------------
        if self.settings.get("use_compression"):
            self._run("setcompress", "0")
            self._log("Rice 16 compression disabled before stacking", LogColor.BLUE)

        # -- Stacking ------------------------------------------------------
        _emit(ProcessingProgress.STACKING, _chunk_label("Stacking"))
        try:
            self._stack(stack_seq, sigma_low, sigma_high, "result")
        except Exception as exc:
            self._log(f"Stacking failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)
            return

        # -- Finalization --------------------------------------------------
        _emit(ProcessingProgress.FINALIZATION, _chunk_label("Finalizing"))
        if chunk_idx is not None:
            self._run("setcompress", "0")
            return
        try:
            self._finalize()
        except Exception as exc:
            self._log(f"Finalization failed{f' for chunk {chunk_idx}' if chunk_idx else ''}: {exc}", LogColor.SALMON)

    def _calibrate(self, seq_name: str, stack_method: Dict[str, Any]) -> None:
        """Calibrate images with dark frame."""
        master_dark = (
            "../../../masters/dark_stacked"
            if self.settings.get("batch_enabled")
            else "../masters/dark_stacked"
        )
        cmd = ["calibrate", seq_name, f"-dark={master_dark}", "-cc=dark", "-cfa", "-equalize_cfa"]
        if not stack_method.get("use_drizzle"):
            cmd.append("-debayer")

        self.siril.log(" ".join(cmd), LogColor.BLUE)
        if not self._run(*cmd):
            raise RuntimeError("Calibration failed")
        
    def _register_regular(self, seq_name: str, stack_method: Dict[str, Any]) -> None:
        """Fallback registration used when local Gaia is absent or plate solve failed."""
        use_drizzle   = stack_method.get("use_drizzle", False)
        drizzle_scale = stack_method.get("drizzle_scale", 1.0)
        two_pass      = self.settings.get("two_pass", False)

        cmd = ["register", seq_name]
        if use_drizzle:
            cmd += [
                "-drizzle",
                f"-scale={drizzle_scale}",
                f"-pixfrac={stack_method.get('drizzle_pixfrac', 1.0)}",
            ]
        if two_pass and (not use_drizzle or drizzle_scale == 1.0):
            cmd.append("-2pass")

        self.siril.log(" ".join(cmd), LogColor.BLUE)
        if not self._run(*cmd):
            raise RuntimeError("Regular registration failed")

    def _seq_apply_reg(self, seq_name: str, stack_method: Dict[str, Any]) -> None:
        """Apply pre-computed plate-solve registration via seqapplyreg."""
        use_drizzle   = stack_method.get("use_drizzle", False)
        drizzle_scale = stack_method.get("drizzle_scale", 1.0)

        framing = "-framing=max" if use_drizzle else "-framing=cog"

        cmd = ["seqapplyreg", seq_name, "-kernel=square", framing]
        if use_drizzle:
            cmd += [
                "-drizzle",
                f"-scale={drizzle_scale}",
                f"-pixfrac={stack_method.get('drizzle_pixfrac', 1.0)}",
            ]

        try:
            self.siril.log(" ".join(cmd), LogColor.BLUE)
            self.siril.cmd(*cmd)
        except (s.DataError, s.CommandError, s.SirilError) as exc:
            self._log(f"seqapplyreg error: {exc}", LogColor.SALMON)
            self._register_regular(seq_name, stack_method)

    def _stack(self, seq_name: str, sigma_low: float, sigma_high: float, output_name: str) -> None:
        """Stack images."""
        has_registration = "r_pp_" in seq_name
        weight = "-weight=wfwhm" if has_registration else "-weight=noise"

        cmd = [
            "stack", seq_name,
            "rej", str(sigma_low), str(sigma_high),
            "-norm=addscale", "-output_norm",
            weight, "-maximize"
        ]
        if has_registration and not self.settings.get("spcc"):
            cmd.append("-rgb_equal")
        if self.settings.get("feather_enabled") and self.settings.get("feather_px", 0) > 0:
            cmd.append(f"-feather={self.settings['feather_px']}")
        cmd.append(f"-out={output_name}")

        self.siril.log(" ".join(cmd), LogColor.BLUE)
        if not self._run(*cmd):
            raise RuntimeError("Stacking failed — check available memory")

    def _finalize(self) -> None:
        """Finalize the stacked image."""
        try:
            self.siril.cmd("load", "result")
            self.siril.cmd("icc_remove")
            self.siril.cmd("setcompress", "0")
            try:
                exposure = self.siril.get_image_fits_header("LIVETIME")
            except Exception:
                try:
                    exposure = self.siril.get_image_fits_header("EXPTIME")
                except Exception:
                    exposure = None
            self.final_filename = self._build_filename(exposure)
            self.siril.cmd("save", f"../{self.final_filename}")
            self.siril.cmd("cd", "..")
        except Exception as exc:
            raise RuntimeError(f"Finalization failed: {exc}")

    def _seq_plate_solve(self, seq_name: str) -> bool:
        """Plate solve a sequence of images."""
        try:
            if not hasattr(self, 'simbad_coordinates'):
                self._log("No SIMBAD coordinates available - skipping sequence plate solve", LogColor.SALMON)
                return False

            ra_deg, dec_deg = self.simbad_coordinates
            focal_length = self.settings.get('focal_length_mm', 250.0)
            pixel_size = self.settings.get('pixel_size_um', 2.9)

            cmd = [
                "seqplatesolve",
                seq_name,
                f"{ra_deg},{dec_deg}",
                f"-focal={focal_length}",
                f"-pixelsize={pixel_size}",
                "-nocache",
                "-force",
                "-disto=ps_distortion",
                "-order=1",
                "-radius=20",
            ]

            self.siril.log(" ".join(cmd), LogColor.BLUE)
            if not self._run(*cmd):
                self._log(f"Sequence plate solve failed for {seq_name}", LogColor.SALMON)
                return False

            return True

        except Exception as exc:
            self._log(f"Error during sequence plate solve: {exc}", LogColor.RED)
            return True

    def _run_background_extraction(self, seq_name: Optional[str] = None) -> None:
        """Extract background from images."""
        if seq_name:
            cmd = [
                "seqsubsky", seq_name,
                "1",
                "-samples=40",
            ]
        else:
            cmd = [
                "subsky", "1", 
                "-samples=40", 
            ]
        self.siril.log(" ".join(cmd), LogColor.BLUE)
        if not self._run(*cmd):
            raise RuntimeError("Background Extraction failed")

    @staticmethod
    def _prepare_chunks(src: Path, dest_root: Path, batch_size: int) -> None:
        
        if not src.exists():
            self._log(f"Source directory {src} does not exist; skipping chunking.", LogColor.SALMON)
            return
    
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
            if len(batch) == 1:
                only = lights_dir / batch[0].name
                shutil.copy2(only, lights_dir / f"{batch[0].stem}_dup{batch[0].suffix}")

    def _process_batch_sessions(
        self,
        stack_method: Dict[str, Any],
        sigma_low: float,
        sigma_high: float,
    ) -> None:
        batch_size = int(self.settings.get("batch_size", 100))
        saved = {k: self.settings[k] for k in ("two_pass", "feather_enabled")}
        self.settings.update({"two_pass": False, "feather_enabled": False})
        intermediate_method = (
            STACKING_METHODS["Bayer Drizzle (Recommended)"]
            if stack_method.get("use_drizzle") and stack_method.get("drizzle_scale", 1.0) > 1.0
            else stack_method
        )
        try:
            self._prepare_chunks(Path(self.workdir, "process"), Path(self.workdir, "Temp"), batch_size)
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
                if not self._run("link", "light", "-out=../process"):
                    continue
                if not self._run("cd", f'"{sess}/process"'):
                    continue
                self.light_seq_name = "light"
                self._log("─" * 48, LogColor.BLUE)
                self._log(f"  Processing chunk {idx} of {total_chunks}", LogColor.BLUE)
                self._log("─" * 48, LogColor.BLUE)
                self._process_standard(intermediate_method, sigma_low, sigma_high,
                                       chunk_idx=idx, total_chunks=total_chunks,
                                       active_process_dir=str(sess / "process"))
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
            self.siril.cmd("close")
        self.settings.update(saved)
        self.progress.emit(ProcessingProgress.FINALIZATION, "Finalizing batch")
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

    def _stack_final(self, stack_method: Dict[str, Any]) -> None:
        final_dir   = Path(self.workdir) / "final_stack"
        chunk_files = sorted(final_dir.glob("pp_session_*.fits"))

        if not chunk_files:
            self._log("No chunk results found in final_stack/ — skipping final combine.", LogColor.SALMON)
            return

        if len(chunk_files) == 1:
            try:
                self.siril.cmd("cd", f'"{str(final_dir)}"')
                self.siril.cmd("load", chunk_files[0].name)
                self.final_filename = self._build_filename(None, batch=True)
                self.siril.cmd("close")

                resolved = Path(self.workdir) / self.final_filename
                shutil.move(str(chunk_files[0]), str(resolved))
                self._log(f"Single chunk — promoted directly to {self.final_filename}", LogColor.BLUE)
            except Exception as exc:
                self._log(f"Could not promote single chunk: {exc}", LogColor.RED)
            return

        use_drizzle = stack_method.get("use_drizzle", False)
        framing = "-framing=max" if use_drizzle else "-framing=cog"

        try:
            process_dir = final_dir / "process"
            process_dir.mkdir(parents=True, exist_ok=True)
            self.siril.cmd("cd", f'"{final_dir.as_posix()}"')
            self.siril.cmd("convert", "pp_session", "-out=./process")
            self.siril.cmd("cd", "process")

            # Plate solve the final sequence if coordinates are available.
            plate_solve_ok = False
            if hasattr(self, "simbad_coordinates") and self.local_catalog_gaia_available:
                ra_deg, dec_deg = self.simbad_coordinates
                try:
                    self.siril.cmd(
                        "seqplatesolve", "pp_session",
                        f"{ra_deg},{dec_deg}",
                        f"-focal={self.settings.get('focal_length_mm', 250.0)}",
                        f"-pixelsize={self.settings.get('pixel_size_um', 2.9)}",
                        "-force",
                    )
                    plate_solve_ok = True
                    self._log("Plate solve on final sequence succeeded", LogColor.BLUE)
                except Exception as exc:
                    self._log(f"Final sequence plate solve failed (non-fatal): {exc}", LogColor.SALMON)

            if plate_solve_ok:
                reg_cmd = ["seqapplyreg", "pp_session", "-kernel=square", framing]
                if not self._run(*reg_cmd):
                    self._log("Registration failed – skipping final stack", LogColor.SALMON)
                    return
                if self.settings.get("two_pass"):
                    if not self._run("seqapplyreg", "pp_session", "-kernel=square", framing):
                        raise RuntimeError("2-pass registration failed for final stack")
            else:
                # No plate solve — fall back to regular registration.
                reg_cmd = ["register", "pp_session"]
                if self.settings.get("two_pass") and not use_drizzle:
                    reg_cmd.append("-2pass")
                if not self._run(*reg_cmd):
                    self._log("Registration failed – skipping final stack", LogColor.SALMON)
                    return

            stack_cmd = ["stack", "r_pp_session", "rej", "3", "3",
                         "-norm=addscale", "-output_norm", "-weight=wfwhm", "-maximize",
                         "-out=final_stacked_batch.fit"]
            if not self._run(*stack_cmd):
                self._log("Stacking failed – final stack incomplete", LogColor.SALMON)
                return


            self.final_filename = self._build_filename(None, batch=True)
            self.siril.cmd("close")

            resolved = Path(self.workdir) / self.final_filename
            shutil.move(str(process_dir / "final_stacked_batch.fit"), str(resolved))
            self._log(f"Final stacked image written to {self.final_filename}", LogColor.BLUE)
        except Exception as exc:
            self._log(f"Final stack failed: {exc}", LogColor.SALMON)
    
    def _run_spcc(self) -> None:
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
        else:
            cmd = f'spcc "-oscsensor={sensor}" "-rfilter=NoFilter" "-gfilter=NoFilter" "-bfilter=NoFilter"'
        self.siril.log(f"Running SPCC: {cmd}", LogColor.BLUE)
        self.siril.cmd(cmd)
        
    def _run_autostretch(self) -> None:
        """Apply auto stretch to the image."""
        try:
            self.siril.cmd(
                "autostretch",
                "-linked",
                str(self.settings.get("shadowsclip", -2.8)),
                str(self.settings.get("targetbg", 0.25))
            )
        except Exception as exc:
            self._log(f"Auto-stretch failed: {exc}", LogColor.SALMON)

    def _run_poststacking(self) -> None:
        """Run all post-stacking operations with per-step progress."""
        total_steps  = max(sum([
            bool(self.settings.get("spcc")),
            bool(self.settings.get("autostretch")),
        ]), 1)
        current_step = 0

        try:
            if self.settings.get("batch_enabled"):
                self.siril.cmd("cd", f'"{self.workdir}"')
                load_name = self.final_filename if self.final_filename else "final_stacked_batch.fit"
                if not self._run("load", load_name):
                    return
            else:
                self.siril.cmd("cd", f'"{self.workdir}"')
                if hasattr(self, 'final_filename') and self.final_filename:
                    if not self._run("load", self.final_filename):
                        self._log("Could not load final stacked image", LogColor.SALMON)
                        return
                else:
                    self._log("No final image filename available", LogColor.SALMON)
                    return
        except Exception as exc:
            self._log(f"Post-stacking load failed: {exc}", LogColor.SALMON)
            return

        if self.settings.get("spcc"):
            current_step += 1
            self.progress.emit(
                int(current_step / total_steps * 100), "Plate solving"
            )
            if hasattr(self, "simbad_coordinates"):
                ra_deg, dec_deg = self.simbad_coordinates
                focal_length = self.settings.get('focal_length_mm', 250.0)
                pixel_size   = self.settings.get('pixel_size_um', 2.9)

                if self.settings.get("stacking_method") == "Drizzle 2x Upscale":
                    focal_length *= 2
                    self._log(
                        f"Drizzle 2× Upscale detected – using effective focal length "
                        f"{focal_length:.1f} mm for plate solving",
                        LogColor.BLUE,
                    )

                cmd = [
                    "platesolve",
                    f"{ra_deg:.12f},{dec_deg:.12f}",
                    f"-focal={focal_length}",
                    f"-pixelsize={pixel_size}",
                    "-force",
                ]
                self.siril.log(" ".join(cmd), LogColor.BLUE)
                plate_solve_ok = self._run(*cmd)
                if not plate_solve_ok:
                    self._log("Plate solve failed for final image", LogColor.SALMON)
            else:
                self._log("No SIMBAD coordinates available – skipping plate solve", LogColor.SALMON)
                plate_solve_ok = False

            if plate_solve_ok:
                current_step += 1
                self.progress.emit(
                    int(current_step / total_steps * 100), "Color calibrating"
                )
                self._run_spcc()
            else:
                self.siril.log("Plate solving failed – skipping SPCC", LogColor.SALMON)

        if self.settings.get("autostretch"):
            current_step += 1
            self.progress.emit(
                int(current_step / total_steps * 100), "Auto‑stretching"
            )
            self._run_autostretch()

        if self.settings.get("spcc") or self.settings.get("autostretch"):
            save_name = Path(self.workdir) / (self.final_filename or "final_stacked_batch.fit")
            self._run("save", f'"{save_name.with_suffix("")}"')

    def _log_summary(self, elapsed_sec: float) -> None:
        """Log processing summary."""
        s = self.settings
        mins, secs = divmod(int(elapsed_sec), 60)
        self._log("─" * 48, LogColor.BLUE)
        self._log("  Processing Summary", LogColor.BLUE)
        self._log("─" * 48, LogColor.BLUE)
        self._log(f"  Sky Quality    : {s.get('sky_quality', '—')}", LogColor.BLUE)
        self._log(f"  Method         : {s.get('stacking_method', '—')}", LogColor.BLUE)
        self._log(f"  BGE            : {'Yes' if s.get('bge') else 'No'}", LogColor.BLUE)
        self._log(f"  2-Pass         : {'Yes' if s.get('two_pass') else 'No'}", LogColor.BLUE)
        self._log(f"  Compression    : {'Yes' if s.get('use_compression') else 'No'}", LogColor.BLUE)
        self._log(f"  Feathering     : {s.get('feather_px', 0)} px" if s.get('feather_enabled') else "  Feathering     : No", LogColor.BLUE)
        self._log(f"  Batch          : {'Yes — ' + str(s.get('batch_size')) + ' img/chunk' if s.get('batch_enabled') else 'No'}", LogColor.BLUE)
        self._log(f"  SPCC           : {'Yes (' + s.get('spcc_filter', 'No Filter') + ')' if s.get('spcc') else 'No'}", LogColor.BLUE)
        self._log(f"  Auto-Stretch   : {'Yes' if s.get('autostretch') else 'No'}", LogColor.BLUE)
        self._log(f"  Output         : {self.final_filename or 'final_stacked_batch.fit'}", LogColor.BLUE)
        self._log(f"  Processing Time: {mins}m {secs:02d}s", LogColor.GREEN)
        self._log("─" * 48, LogColor.BLUE)

    def run(self) -> None:
        """QThread entry point."""
        try:
            self._process()
        except Exception as exc:
            self.finished.emit(False, f"Error: {exc}")

    def _build_filename(self, exposure: Optional[Any], batch: bool = False) -> str:
        """Build output filename from active settings."""
        s = self.settings
        ts = datetime.now().strftime("%H%M")
        parts: List[str] = []

        obj_name = None
        try:
            header_dict = self.siril.get_image_fits_header(return_as='dict')
            if isinstance(header_dict, dict):
                obj_name = str(header_dict.get("OBJECT", "")).strip() or None
        except Exception:
            pass
        self._log(f"Output filename — OBJECT header: {obj_name!r}", LogColor.BLUE)

        if isinstance(obj_name, str) and obj_name.strip():
            base = re.sub(r"[^\w\-]", "", obj_name.strip().replace(" ", "_"))
            if not base:
                base = "result"
        else:
            base = "result"

        if exposure is not None:
            try:
                parts.append(f"{int(float(exposure))}s")
            except (ValueError, TypeError):
                pass

        if s.get("bge"):
            parts.append("bge")
        if s.get("two_pass"):
            parts.append("2pass")
        if s.get("use_compression"):
            parts.append("rice")
        if s.get("feather_enabled") and s.get("feather_px", 0) > 0:
            parts.append(f"f{s['feather_px']}px")
        if s.get("spcc"):
            filter_tag = {
                "Dual Band Ha/Oiii": "dualband",
                "City Light Pollution": "cls",
            }.get(s.get("spcc_filter", ""), "spcc")
            parts.append(filter_tag)
        if s.get("autostretch"):
            parts.append("stretch")

        suffix = ("_" + "_".join(parts)) if parts else ""

        return f"{base}{suffix}_{ts}.fit"

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class VesperaProGUI(QDialog):
    """Main preprocessing dialog for Vespera observations."""

    def __init__(self, siril: Any, app: QApplication) -> None:
        super().__init__()
        self.siril = siril
        self.app = app
        self.worker: Optional[ProcessingThread] = None
        self.workdir = ""
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx    = 0
        self._spinner_base   = ""
        self._spinner_timer  = QTimer(self)
        self._spinner_timer.setInterval(100)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        import json
        self._settings_file = Path(os.path.expanduser("~")) / ".vespera_preprocessing.json"
        self.current_settings: Dict[str, Any] = {}
        self.folder_structure: Optional[str] = None

        self.setFixedSize(560, 830)
        self.setWindowTitle(f"Vespera — Preprocessing v{VERSION}")
        self.setStyleSheet(DARK_STYLESHEET)

        # Detect local Gaia catalogue
        self.local_catalog_gaia_available = False
        try:
            gaia_path = self.siril.get_siril_config("core", "catalogue_gaia_astro")
            if gaia_path and gaia_path != "(not set)" and os.path.isfile(gaia_path):
                self.local_catalog_gaia_available = True
        except Exception:
            pass

        self._setup_ui()
        self._load_settings()
        self._check_folders()

    def _start_processing(self) -> None:
        """Start the processing pipeline in a background thread."""
        self._save_settings()
        self._set_gui_enabled(False)
        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Processing")
        self.status.setStyleSheet("color: #ffcc44;")
        self.log_area.clear()

        temp_dir = Path(self.siril.get_siril_wd()) / "Temp"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                self._log("Previous 'Temp' directory cleaned", LogColor.BLUE)
            except Exception as exc:
                self._log(f"Could not delete 'Temp' folder: {exc}", LogColor.RED)

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

            self.worker = ProcessingThread(
                self.siril, workdir, settings,
                self.folder_structure,
                self.local_catalog_gaia_available,
            )
            self.worker.log_area = self.log_area
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.worker.log.connect(self._log)
            self.worker.start()

        except Exception as exc:
            self._log(f"Start error: {exc}", LogColor.RED)
            self._set_gui_enabled(True)
            self.btn_start.setEnabled(True)

    # Steps that can take many minutes — activate indeterminate + spinner
    _LONG_STEPS = (
        "Calibrat", "Background Extract", "Resolving coord",
        "Plate solv", "Registering", "Stacking", "Finalizing",
    )

    def _on_progress(self, percent: int, message: str) -> None:
        """Update progress bar and status; spin on long steps."""
        self.progress.setValue(percent)
        is_long = any(kw in message for kw in self._LONG_STEPS)
        if is_long:
            if not self._spinner_timer.isActive():
                self._start_spinner(message)
            else:
                # Step changed — update base label, spinner keeps running
                self._spinner_base = message
        else:
            self._stop_spinner(percent, message)
        QApplication.processEvents()  # Keep GUI responsive

    def _on_finished(self, success: bool, message: str) -> None:
        """Handle processing completion."""
        self._stop_spinner(100, "")
        self._set_gui_enabled(True)
        self.btn_start.setEnabled(True)
        if hasattr(self, "worker") and self.worker:
            self._write_console_log()
        if success:
            self.status.setText(f"✓ {message}")
            self.status.setStyleSheet("color: #55cc77;")
            try:
                self.siril.log("Stacking Complete!", color=LogColor.GREEN)
            except Exception:
                pass
        else:
            self.status.setText(f"✗ {message}")
            self.status.setStyleSheet("color: #cc5555;")
            self._log(f"FAILED: {message}", LogColor.RED)

    def _write_console_log(self) -> None:
        """Write console output to log file."""
        if not hasattr(self, "worker") or not self.worker:
            return
        logs_dir = Path(self.workdir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"siril_console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        try:
            with open(log_file, "w", encoding="utf-8") as fh:
                fh.write(self._config_summary() + "\n")
                if hasattr(self.worker, 'console_messages'):
                    fh.writelines(msg + "\n" for msg in self.worker.console_messages)
        except Exception as exc:
            self._log(f"Failed to write console log: {exc}", LogColor.RED)

    def _config_summary(self) -> str:
        """Generate configuration summary for logging."""
        s = self.current_settings
        return (
            f"Stacking method: {s.get('stacking_method', 'Unknown')}, "
            f"Compression: {'Yes' if s.get('use_compression') else 'No'}, "
            f"Feathering: {'Yes' if s.get('feather_enabled') else 'No'}, "
            f"2‑Pass: {'Yes' if s.get('two_pass') else 'No'}, "
            f"Batch: {'Yes' if s.get('batch_enabled') else 'No'}, "
            f"Chunks: {getattr(self, 'num_chunks', 1)}"
        )
    
    def _tick_spinner(self) -> None:
        """Advance the spinner one frame."""
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        frame = self._spinner_frames[self._spinner_idx]
        self.status.setText(f"{self._spinner_base}  {frame}")

    def _start_spinner(self, base_msg: str) -> None:
        """Start braille spinner on status label."""
        self._spinner_base = base_msg
        self._spinner_idx  = 0
        self.status.setText(base_msg)
        self._spinner_timer.start()

    def _stop_spinner(self, percent: int, msg: str) -> None:
        """Stop spinner and update status label."""
        self._spinner_timer.stop()
        self.progress.setValue(percent)
        self.status.setText(msg)

    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        if self.log_area:
            append_colored_text(self.log_area, msg, color)

    def _set_gui_enabled(self, enabled: bool) -> None:
        """Lock or unlock all GUI controls."""
        # Dropdowns
        self.combo_sky.setEnabled(enabled)
        self.combo_stack.setEnabled(enabled)
        
        # Buttons
        self.btn_browse_dir.setEnabled(enabled)
        
        # Checkboxes
        self.chk_bg_extract.setEnabled(enabled)
        self.chk_feather.setEnabled(enabled)
        self.chk_two_pass.setEnabled(enabled)
        self.chk_compression.setEnabled(enabled)
        self.chk_clean_temp.setEnabled(enabled)
        self.chk_batch.setEnabled(enabled)
        self.spcc_cb.setEnabled(enabled)
        self.autostretch_cb.setEnabled(enabled)
        
        # Spinboxes
        self.feather_slider.setEnabled(enabled)
        self.spin_batch_size.setEnabled(enabled)
        
        # Combos
        self.spcc_filter_combo.setEnabled(enabled)
    
    def _setup_ui(self) -> None:
        """Initialize all UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Main tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_main_tab(), "Pipeline")
        tabs.addTab(self._create_info_tab(), "Reference")
        layout.addWidget(tabs)

        # Progress group
        progress_group = QGroupBox("Progress")
        progress_group.setFixedHeight(220)
        progress_layout = QVBoxLayout(progress_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_layout.addWidget(self.progress)

        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(100)
        self.log_area.setMaximumHeight(100)
        progress_layout.addWidget(self.log_area)
        layout.addWidget(progress_group)

        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_start = QPushButton("▶  Run Pipeline")
        self.btn_start.setObjectName("start")
        self.btn_start.clicked.connect(self._start_processing)
        btn_layout.addWidget(self.btn_start)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _create_main_tab(self) -> QWidget:
        """Create the main pipeline configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # INPUT section
        input_group = QGroupBox("INPUT")
        input_group.setObjectName("stage")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(4)

        wd_row = QHBoxLayout()
        self.lbl_workdir = QLabel("No directory selected")
        self.lbl_workdir.setObjectName("hint")
        self.lbl_workdir.setWordWrap(True)
        self.btn_browse_dir = QPushButton("Browse…")
        self.btn_browse_dir.setObjectName("browse")
        self.btn_browse_dir.clicked.connect(self._browse_working_directory)
        wd_row.addWidget(self.lbl_workdir, 1)
        wd_row.addWidget(self.btn_browse_dir)
        input_layout.addLayout(wd_row)

        counts_row = QHBoxLayout()
        self.lbl_darks = QLabel("Darks: —")
        self.lbl_lights = QLabel("Lights: —")
        self.lbl_darks.setObjectName("hint")
        self.lbl_lights.setObjectName("hint")
        counts_row.addWidget(self.lbl_darks)
        counts_row.addWidget(QLabel("·"))
        counts_row.addWidget(self.lbl_lights)
        counts_row.addStretch()
        self.lbl_structure = QLabel("")
        self.lbl_structure.setObjectName("hint")
        counts_row.addWidget(self.lbl_structure)
        input_layout.addLayout(counts_row)
        layout.addWidget(input_group)

        # PRESETS section
        prst_group = QGroupBox("PRESETS")
        prst_group.setObjectName("stage")
        prst_layout = QVBoxLayout(prst_group)
        prst_layout.setSpacing(16)

        sky_row = QHBoxLayout()
        sky_label = QLabel("Sky Quality:")
        sky_label.setFixedWidth(120)
        self.combo_sky = QComboBox()
        for name in SKY_PRESETS:
            self.combo_sky.addItem(name)
        self.combo_sky.currentTextChanged.connect(self._on_sky_changed)
        self.combo_sky.setFixedWidth(240)

        sky_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.combo_sky.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        sky_row.addWidget(sky_label, 1)
        sky_row.addSpacing(8)
        sky_row.addWidget(self.combo_sky, 2)
        self.lbl_sigma = QLabel("σ —/—")
        self.lbl_sigma.setObjectName("sigma")
        self.lbl_sigma.setToolTip("Sigma-low / Sigma-high rejection thresholds")
        sky_row.addWidget(self.lbl_sigma)
        prst_layout.addLayout(sky_row)

        chk_row = QHBoxLayout()
        chk_row.setSpacing(20)

        left_col = QVBoxLayout()
        self.chk_compression = QCheckBox("RICE 16 Compression")
        self.chk_compression.setToolTip("Apply Rice 16-bit lossless compression...")
        left_col.addWidget(self.chk_compression)

        right_col = QVBoxLayout()
        self.chk_clean_temp = QCheckBox("Clean Temp Files")
        self.chk_clean_temp.setToolTip("Delete process/ masters/ final_stack/...")
        right_col.addWidget(self.chk_clean_temp)

        chk_row.addLayout(left_col)
        chk_row.addLayout(right_col)
        prst_layout.addLayout(chk_row)

        prst_layout.addSpacing(10)
        layout.addWidget(prst_group)

        # REGISTRATION / STACKING section
        stack_group = QGroupBox("REGISTRATION / STACKING")
        stack_group.setObjectName("stage")
        stack_layout = QVBoxLayout(stack_group)
        stack_layout.setSpacing(16)

        method_row = QHBoxLayout()
        method_label = QLabel("Stacking Method:")
        method_label.setFixedWidth(120)
        self.combo_stack = QComboBox()
        for idx, (name, cfg) in enumerate(STACKING_METHODS.items()):
            self.combo_stack.addItem(name)
            if "tooltip" in cfg:
                self.combo_stack.setItemData(idx, cfg["tooltip"], Qt.ItemDataRole.ToolTipRole)
        self.combo_stack.currentTextChanged.connect(self._on_stack_changed)
        self.combo_stack.setFixedWidth(240)

        method_row.addWidget(method_label)
        method_row.addSpacing(8)
        method_row.addWidget(self.combo_stack)

        empty_widget = QLabel("")
        empty_widget.setObjectName("sigma")
        empty_widget.setFixedSize(60, 20)
        method_row.addStretch()
        method_row.addWidget(empty_widget)

        stack_layout.addLayout(method_row)

        self.lbl_stack_desc = QLabel("")
        self.lbl_stack_desc.setObjectName("hint")
        self.lbl_stack_desc.setWordWrap(True)
        stack_layout.addWidget(self.lbl_stack_desc)

        stack_layout.addWidget(make_hline())

        opts_row = QHBoxLayout()
        opts_row.setSpacing(16)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        self.chk_bg_extract = QCheckBox("Background Extraction")
        self.chk_bg_extract.setToolTip(
            "Background subtraction on calibrated debayered frames, before registration.\n"
            "⚠ Incompatible with all Drizzle methods (drizzle requires CFA input).\n"
            "Only available with Standard Registration."
        )
        left_col.addWidget(self.chk_bg_extract)

        self.chk_two_pass = QCheckBox("2-Pass Registration")
        self.chk_two_pass.setToolTip(
            "Second alignment pass with -framing=max for sessions\n"
            "with significant field rotation. Needs ≥50 frames/chunk."
        )
        left_col.addWidget(self.chk_two_pass)

        feather_row = QHBoxLayout()
        self.chk_feather = QCheckBox("Feathering")
        self.chk_feather.setToolTip("Blend stacked frame edges (0–50 px) to hide seams")
        self.feather_slider = QSpinBox()
        self.feather_slider.setRange(0, 50)
        self.feather_slider.setValue(0)
        self.feather_slider.setSuffix(" px")
        self.feather_slider.setFixedWidth(82)
        feather_row.addWidget(self.chk_feather)
        feather_row.addWidget(self.feather_slider)
        right_col.addLayout(feather_row)

        batch_row = QHBoxLayout()
        self.chk_batch = QCheckBox("Batch")
        self.chk_batch.setToolTip(
            "Split lights into chunks to reduce peak disk usage.\n"
            "Min recommended: 20 frames/chunk."
        )
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(20, 2048)
        self.spin_batch_size.setValue(200)
        self.spin_batch_size.setSuffix(" img")
        self.spin_batch_size.setFixedWidth(82)
        batch_row.addWidget(self.chk_batch)
        batch_row.addWidget(self.spin_batch_size)
        right_col.addLayout(batch_row)

        opts_row.addLayout(left_col)
        opts_row.addLayout(right_col)
        stack_layout.addLayout(opts_row)
        layout.addWidget(stack_group)

        # POST-STACKING section
        post_group = QGroupBox("POST-STACKING")
        post_group.setObjectName("stage")
        post_layout = QVBoxLayout(post_group)

        post_opts_row = QHBoxLayout()
        post_opts_row.setSpacing(20)

        left_col = QVBoxLayout()
        self.autostretch_cb = QCheckBox("Autostretch")
        left_col.addWidget(self.autostretch_cb)

        right_col = QVBoxLayout()
        spcc_row = QHBoxLayout()
        self.spcc_cb = QCheckBox("SPCC")
        self.spcc_filter_combo = QComboBox()
        self.spcc_filter_combo.addItems(["No Filter", "Dual Band Ha/Oiii", "City Light Pollution"])
        self.spcc_filter_combo.setFixedWidth(170)

        spcc_row.addWidget(self.spcc_cb)
        spcc_row.addWidget(self.spcc_filter_combo)
        right_col.addLayout(spcc_row)

        post_opts_row.addLayout(left_col)
        post_opts_row.addLayout(right_col)
        post_layout.addLayout(post_opts_row)

        layout.addWidget(post_group)

        return widget

    def _create_info_tab(self) -> QWidget:
        """Create the reference/information tab."""
        widget = QWidget()
        outer = QVBoxLayout(widget)
        outer.setSpacing(6)
        outer.setContentsMargins(6, 6, 6, 6)

        from PyQt6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: #1e1e2e; width: 8px; border-radius: 4px; }"
            "QScrollBar::handle:vertical { background: #444466; border-radius: 4px; }"
        )

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(10)
        container_layout.setContentsMargins(4, 4, 4, 4)

        def make_section(title: str, html_body: str) -> QGroupBox:
            box = QGroupBox(title)
            layout = QVBoxLayout(box)
            lbl = QLabel(html_body)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #9999bb; font-size: 12pt;")
            lbl.setOpenExternalLinks(True)
            layout.addWidget(lbl)
            return box

        container_layout.addWidget(make_section(
            "Quick Start",
            """
            <p>Point the plugin at your observation folder and press <b>Run Pipeline</b>.
            Folder layout and telescope model are auto-detected.</p>
            <table cellspacing="0" cellpadding="2" width="100%">
              <tr><td width="30%"><b>Organised</b></td><td><code>darks/</code> + <code>lights/</code></td></tr>
              <tr><td><b>Native</b></td><td><code>01-images-initial/</code> + <code>*-dark.fits</code> in root</td></tr>
              <tr><td><b>Flat</b></td><td>All FITS in one folder — auto-sorted by filename</td></tr>
            </table>
            <p>Output: <code>OBJECT_{suffixes}_{HHMM}.fit</code> in the working directory.</p>
            """
        ))

        container_layout.addWidget(make_section(
            "Stacking Methods",
            """
            <table cellspacing="0" cellpadding="3" width="100%">
              <tr><td width="38%" valign="top"><b>Drizzle Gaussian</b></td>
                  <td>Best for 10–15° field rotation. Default choice.</td></tr>
              <tr><td valign="top"><b>Drizzle Square</b></td>
                  <td>Flux-preserving. Preferred for photometry.</td></tr>
              <tr><td valign="top"><b>Drizzle Nearest</b></td>
                  <td>No interpolation artifacts. Use if checkerboard patterns appear.</td></tr>
              <tr><td valign="top"><b>Standard</b></td>
                  <td>No drizzle. Fast. Good for &lt;30 min / &lt;5° rotation.</td></tr>
              <tr><td valign="top"><b>Drizzle 2× Upscale</b></td>
                  <td>2× native resolution. Needs 50+ well-dithered frames. BGE and 2-Pass disabled.</td></tr>
            </table>
            """
        ))

        container_layout.addWidget(make_section(
            "Options",
            """
            <table cellspacing="0" cellpadding="3" width="100%">
              <tr><td width="38%" valign="top"><b>BGE</b></td>
                  <td>Background subtraction before registration. Standard only — incompatible with Drizzle.</td></tr>
              <tr><td valign="top"><b>2-Pass Registration</b></td>
                  <td>Second alignment pass for sessions with significant field rotation. Needs ≥50 frames.</td></tr>
              <tr><td valign="top"><b>RICE 16</b></td>
                  <td>Lossless compression on intermediates. Disabled automatically before stacking.</td></tr>
              <tr><td valign="top"><b>Feathering</b></td>
                  <td>Soft edge blend 0–50 px. Applied to final result only, not batch chunks.</td></tr>
              <tr><td valign="top"><b>Batch</b></td>
                  <td>Stack in N-frame chunks. Min 20 frames. Final combine uses fixed σ 3/3.</td></tr>
              <tr><td valign="top"><b>SPCC</b></td>
                  <td>Photometric colour calibration via Gaia DR3. Requires internet + OBJECT FITS header.</td></tr>
              <tr><td valign="top"><b>Auto-Stretch</b></td>
                  <td>MTF stretch σ −2.8 / bg 0.25. ⚠ Irreversible — disable for linear workflows.</td></tr>
              <tr><td valign="top"><b>Clean Temp</b></td>
                  <td>Deletes <code>process/</code> <code>masters/</code> <code>final_stack/</code>. Verify result first.</td></tr>
            </table>
            <p style="color:#556688; margin-top:8px;">© G. Trainar (2026) — MIT License</p>
            """
        ))

        container_layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)
        return widget

    def _on_sky_changed(self, name: str) -> None:
        """Update sigma values when sky quality changes."""
        if name in SKY_PRESETS:
            p = SKY_PRESETS[name]
            self.lbl_sigma.setText(f"σ {p['sigma_low']}/{p['sigma_high']}")
            self.lbl_sigma.setToolTip(
                f"{p['description']}\nσ-low: {p['sigma_low']}  σ-high: {p['sigma_high']}"
            )

    def _on_stack_changed(self, name: str) -> None:
        """Update description when stacking method changes."""
        if name in STACKING_METHODS:
            self.lbl_stack_desc.setText(STACKING_METHODS[name]["description"])

        if STACKING_METHODS.get(name, {}).get("use_drizzle"):
            disabled = []
            if self.chk_bg_extract.isChecked():
                self.chk_bg_extract.setChecked(False)
                disabled.append("BGE")
            if name == "Drizzle 2x Upscale":
                if self.chk_two_pass.isChecked():
                    self.chk_two_pass.setChecked(False)
                    disabled.append("2-Pass Registration")
            if disabled:
                self._log(
                    f"Drizzle: {' and '.join(disabled)} disabled "
                    f"(incompatible — drizzle requires CFA input).",
                    LogColor.SALMON,
                )

    def _load_settings(self) -> None:
        """Load saved settings from file."""
        data = {}
        try:
            if self._settings_file.exists():
                with open(self._settings_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
        except Exception:
            pass  # First run or corrupted file — use defaults silently

        def g_str(key, default): return str(data.get(key, default))
        def g_bool(key, default): return bool(data.get(key, default))
        def g_int(key, default): return int(data.get(key, default))

        self.combo_sky.setCurrentText(g_str("sky_quality", "Bortle 4-5 (Rural)"))
        self.combo_stack.setCurrentText(g_str("stacking_method", "Bayer Drizzle (Recommended)"))
        self.chk_bg_extract.setChecked(g_bool("bge", False))
        self.feather_slider.setValue(g_int("feather_px", 15))
        self.chk_feather.setChecked(g_bool("feather_enabled", False))
        self.chk_two_pass.setChecked(g_bool("two_pass", False))
        self.chk_compression.setChecked(g_bool("use_compression", False))
        self.chk_clean_temp.setChecked(g_bool("clean_temp", False))
        self.chk_batch.setChecked(g_bool("batch_enabled", False))
        self.spin_batch_size.setValue(g_int("batch_size", 20))
        self.spcc_cb.setChecked(g_bool("spcc", False))
        self.autostretch_cb.setChecked(g_bool("autostretch", True))
        self.spcc_filter_combo.setCurrentText(g_str("spcc_filter", "No Filter"))
        self._on_sky_changed(self.combo_sky.currentText())
        self._on_stack_changed(self.combo_stack.currentText())

    def _save_settings(self) -> None:
        """Save current settings to file."""
        data = {
            "sky_quality": self.combo_sky.currentText(),
            "stacking_method": self.combo_stack.currentText(),
            "bge": self.chk_bg_extract.isChecked(),
            "feather_px": self.feather_slider.value(),
            "feather_enabled": self.chk_feather.isChecked(),
            "two_pass": self.chk_two_pass.isChecked(),
            "use_compression": self.chk_compression.isChecked(),
            "clean_temp": self.chk_clean_temp.isChecked(),
            "batch_enabled": self.chk_batch.isChecked(),
            "batch_size": self.spin_batch_size.value(),
            "spcc": self.spcc_cb.isChecked(),
            "autostretch": self.autostretch_cb.isChecked(),
            "spcc_filter": self.spcc_filter_combo.currentText(),
        }
        try:
            with open(self._settings_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            self._log(f"Could not save settings: {exc}", LogColor.SALMON)

    def _browse_working_directory(self) -> None:
        """Open file dialog to select working directory."""
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
        """Check and update folder status."""
        try:
            workdir = self.siril.get_siril_wd()
            self.workdir = workdir
            display_path = workdir if len(workdir) <= 48 else "…" + workdir[-46:]
            self.lbl_workdir.setText(display_path)
            self.lbl_workdir.setToolTip(workdir)

            darks_dir = os.path.join(workdir, "darks")
            lights_dir = os.path.join(workdir, "lights")

            num_darks_org = count_fits_in(darks_dir) if os.path.exists(darks_dir) else 0
            num_lights_org = count_fits_in(lights_dir) if os.path.exists(lights_dir) else 0

            native = self._detect_native_structure(workdir)

            if num_darks_org > 0 and num_lights_org > 0:
                self.folder_structure = "organized"
                num_darks, num_lights = num_darks_org, num_lights_org
                self._set_structure_label("darks/ · lights/")
            elif native:
                self.folder_structure = "native"
                num_darks = native["num_darks"]
                num_lights = native["num_lights"]
                self._set_structure_label("native Vespera")
            elif self._organise_flat_directory(workdir):
                self.folder_structure = "organized"
                num_darks = count_fits_in(os.path.join(workdir, "darks"))
                num_lights = count_fits_in(os.path.join(workdir, "lights"))
                self._set_structure_label("flat → reorganised")
            else:
                self.folder_structure = None
                num_darks = num_lights = 0
                self._set_structure_label("not detected", error=True)

            self._update_count_label(self.lbl_darks, "Darks", num_darks)
            self._update_count_label(self.lbl_lights, "Lights", num_lights)
            self.btn_start.setEnabled(num_darks > 0 and num_lights > 0)

        except Exception as exc:
            self._log(f"Error: {exc}")
            self.btn_start.setEnabled(False)

    def _set_structure_label(self, text: str, error: bool = False) -> None:
        """Set folder structure label with optional error styling."""
        self.lbl_structure.setText(text)
        self.lbl_structure.setStyleSheet(
            "color: #cc5555; font-size: 9pt;" if error else "color: #445588; font-size: 9pt;"
        )

    def _update_count_label(self, label: QLabel, kind: str, count: int) -> None:
        """Update frame count label with appropriate styling."""
        if count > 0:
            label.setText(f"✓ {kind}: {count}")
            label.setStyleSheet("color: #55aa77; font-size: 9pt; font-family: 'Menlo', 'Courier New';")
        else:
            label.setText(f"✗ {kind}: not found")
            label.setStyleSheet("color: #aa4444; font-size: 9pt; font-family: 'Menlo', 'Courier New';")

    def _detect_native_structure(self, workdir: str) -> Optional[Dict[str, Any]]:
        """Detect native Vespera folder structure."""
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
                "dark_files": dark_files,
                "light_files": light_files,
                "num_darks": len(dark_files),
                "num_lights": len(light_files),
                "images_initial": images_initial,
            }
        return None

    def _organise_flat_directory(self, workdir: str) -> bool:
        """Organize flat directory into darks/ and lights/ folders."""
        fits_files = glob.glob(os.path.join(workdir, "*.fits"))
        if not fits_files:
            return False
        dark_patterns = ("master_dark.fits", "master-dark.fits", "*dark*.fits")
        dark_files = [f for f in fits_files if any(
            glob.fnmatch.fnmatch(os.path.basename(f).lower(), p) for p in dark_patterns
        )]
        light_files = [f for f in fits_files if f not in dark_files]
        if not dark_files or not light_files:
            return False
        lights_dir = os.path.join(workdir, "lights")
        darks_dir = os.path.join(workdir, "darks")
        os.makedirs(lights_dir, exist_ok=True)
        os.makedirs(darks_dir, exist_ok=True)
        for src in light_files:
            shutil.move(src, os.path.join(lights_dir, os.path.basename(src)))
        for src in dark_files:
            shutil.move(src, os.path.join(darks_dir, os.path.basename(src)))
        self._log(
            f"Organised flat directory: {len(light_files)} light(s) → lights/, "
            f"{len(dark_files)} dark(s) → darks/",
            LogColor.GREEN,
        )
        return True

    def _build_settings(self) -> Dict[str, Any]:
        """Build settings dictionary from UI state."""
        return {
            "sky_quality": self.combo_sky.currentText(),
            "stacking_method": self.combo_stack.currentText(),
            "bge": self.chk_bg_extract.isChecked(),
            "feather_px": self.feather_slider.value(),
            "feather_enabled": self.chk_feather.isChecked(),
            "two_pass": self.chk_two_pass.isChecked(),
            "use_compression": self.chk_compression.isChecked(),
            "clean_temp": self.chk_clean_temp.isChecked(),
            "batch_enabled": self.chk_batch.isChecked(),
            "batch_size": self.spin_batch_size.value(),
            "spcc": self.spcc_cb.isChecked(),
            "autostretch": self.autostretch_cb.isChecked(),
            "focal_length_mm": 250.0,
            "pixel_size_um": 2.9,
            "spcc_sensor": "Sony IMX585",
            "spcc_filter": self.spcc_filter_combo.currentText(),
        }

    def _config_summary(self) -> str:
        """Generate configuration summary for logging."""
        s = self.current_settings
        return (
            f"Stacking method: {s.get('stacking_method', 'Unknown')}, "
            f"Compression: {'Yes' if s.get('use_compression') else 'No'}, "
            f"Feathering: {'Yes' if s.get('feather_enabled') else 'No'}, "
            f"2‑Pass: {'Yes' if s.get('two_pass') else 'No'}, "
            f"Batch: {'Yes' if s.get('batch_enabled') else 'No'}, "
            f"Chunks: {getattr(self, 'num_chunks', 1)}"
        )

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
            if _global_gui and hasattr(_global_gui, "worker") and _global_gui.worker:
                _global_gui._write_console_log()
            sys.__excepthook__(exc_type, exc_value, tb)

        sys.excepthook = _crash_handler
        gui.show()
        app.exec()

    except Exception as exc:
        print(f"Error: {exc}")

if __name__ == "__main__":
    main()