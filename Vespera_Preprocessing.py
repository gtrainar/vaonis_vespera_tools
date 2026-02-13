##############################################
# Vespera — Preprocessing
# Automated Stacking for Alt‑Az Mounts
# Authors: Claude (Anthropic) (2025)
#          G. Trainar
# Contact: github.com/gtrainar
##############################################
# (c) 2025 Claude (Anthropic), G. Trainar - MIT License
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
from typing import Optional, Dict, Any, List
from pathlib import Path
import re
import time                     
from datetime import datetime

try:
    import sirilpy as s
    from sirilpy import LogColor
except ImportError:
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

# Ensure dependencies
s.ensure_installed("PyQt6")

from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QProgressBar, QMessageBox,
                             QTextEdit, QGroupBox, QComboBox, QCheckBox,
                             QSpinBox, QDoubleSpinBox, QTabWidget, QWidget,
                             QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont

VERSION = "1.1.0"

CHANGELOG = """
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

# Constants for processing progress percentages
class ProcessingProgress:
    CLEANUP = 5
    DARK_PROCESSING = 10
    LIGHT_CONVERSION = 20
    CALIBRATION = 30
    REGISTRATION = 50
    STACKING = 75
    FINALIZATION = 88
    COMPLETE = 100

# Sky Quality Presets (Bortle scale)
SKY_PRESETS = {
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

# Stacking methods with tooltips explaining technical details
STACKING_METHODS = {
    "Bayer Drizzle (Recommended)": {"description": "Best for field rotation, gaussian kernel for smooth CFA",
                                    "tooltip": ("Uses Gaussian drizzle kernel with area‑based interpolation.\n\n"
                                                "• Gaussian kernel: Produces smooth, centrally‑peaked PSFs\n"
                                                "• Area interpolation: Reduces moiré patterns from field rotation\n"
                                                "• Best choice for typical Vespera sessions with 10‑15° rotation\n\n"
                                                "Technical: scale=1.0, pixfrac=1.0, kernel=gaussian, interp=area"),
                                    "use_drizzle": True,
                                    "drizzle_scale": 1.0,
                                    "drizzle_pixfrac": 1.0,
                                    "drizzle_kernel": "gaussian",
                                    "interp": "area",
                                    "feather_px": 0},
    "Bayer Drizzle (Square)": {"description": "Classic drizzle kernel, mathematically flux‑preserving",
                               "tooltip": ("Uses classic square drizzle kernel (original HST algorithm).\n\n"
                                           "• Square kernel: Mathematically flux‑preserving by construction\n"
                                           "• May show subtle grid patterns with significant field rotation\n"
                                           "• Better for photometry applications\n\n"
                                           "Technical: scale=1.0, pixfrac=1.0, kernel=square, interp=area"),
                               "use_drizzle": True,
                               "drizzle_scale": 1.0,
                               "drizzle_pixfrac": 1.0,
                               "drizzle_kernel": "square",
                               "interp": "area",
                               "feather_px": 0},
    "Bayer Drizzle (Nearest)": {"description": "Nearest‑neighbor interpolation to minimize moiré patterns",
                                 "tooltip": ("Uses nearest‑neighbor interpolation to eliminate moiré.\n\n"
                                             "• Nearest interpolation: No interpolation artifacts at CFA boundaries\n"
                                             "• May appear slightly blocky at pixel level\n"
                                             "• Try this if other methods show checkerboard patterns\n\n"
                                             "Technical: scale=1.0, pixfrac=1.0, kernel=gaussian, interp=nearest"),
                                 "use_drizzle": True,
                                 "drizzle_scale": 1.0,
                                 "drizzle_pixfrac": 1.0,
                                 "drizzle_kernel": "gaussian",
                                 "interp": "nearest",
                                 "feather_px": 0},
    "Standard Registration": {"description": "Faster processing, good for short sessions with minimal rotation",
                              "tooltip": ("Standard debayer‑then‑register workflow (no drizzle).\n\n"
                                          "• Faster processing, lower memory usage\n"
                                          "• Works well for sessions under 30 minutes\n"
                                          "• May show field rotation artifacts at image edges\n"
                                          "• Not recommended for sessions with >5° total rotation"),
                              "use_drizzle": False,
                              "feather_px": 0},
    "Drizzle 2x Upscale": {"description": "Doubles resolution, requires many well‑dithered frames (50+)",
                            "tooltip": ("Upscales to 2x resolution using drizzle algorithm.\n\n"
                                        "• Requires 50+ frames with good sub‑pixel dithering\n"
                                        "• Output will be 7072×7072 pixels (vs 3536×3536)\n"
                                        "• Uses square kernel (only valid choice for scale>1)\n"
                                        "• Significantly increased processing time and file sizes\n\n"
                                        "Note: Lanczos kernels cannot be used with scale>1.0\n"
                                        "Technical: scale=2.0, pixfrac=1.0, kernel=square, interp=area"),
                            "use_drizzle": True,
                            "drizzle_scale": 2.0,
                            "drizzle_pixfrac": 1.0,
                            "drizzle_kernel": "square",
                            "interp": "area",
                            "feather_px": 0},
}

# Dark stylesheet for UI
DARK_STYLESHEET = """
QDialog { background-color: #2b2b2b; color: #e0e0e0; }
QTabWidget::pane { border: 1px solid #444444; background-color: #2b2b2b; }
QTabBar::tab {
    background-color: #3c3c3c;
    color: #aaaaaa;
    padding: 8px 16px;
    border: 1px solid #444444;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #2b2b2b; color: #ffffff; }
QTabBar::tab:hover { background-color: #444444; }

QGroupBox {
    border: 1px solid #444444;
    margin-top: 12px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #88aaff;
}

QLabel { color: #cccccc; font-size: 10pt; }
QLabel#title { color: #88aaff; font-size: 14pt; font-weight: bold; }
QLabel#subtitle { color: #888888; font-size: 9pt; }
QLabel#status { color: #ffcc00; font-size: 10pt; }
QLabel#error { color: #ff8888; }
QLabel#info { color: #88aaff; font-size: 9pt; }

QComboBox {
    background-color: #3c3c4c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 10px;
    min-width: 200px;
}
QComboBox:hover { border-color: #88aaff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #aaaaaa;
}
QComboBox QAbstractItemView {
    background-color: #3c3c4c;
    color: #ffffff;
    selection-background-color: #285299;
    border: 1px solid #555555;
}

QCheckBox { color: #cccccc; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #666666;
    background: #3c3c4c;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #285299;
    border: 1px solid #88aaff;
}
QCheckBox::indicator:hover { border-color: #88aaff; }

QSpinBox, QDoubleSpinBox {
    background-color: #3c3c4c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}

QProgressBar {
    border: 1px solid #555555;
    border-radius: 4px;
    background-color: #3c3c4c;
    text-align: center;
    color: #ffffff;
    min-height: 20px;
}
QProgressBar::chunk { background-color: #285299; border-radius: 3px; }

QPushButton {
    background-color: #444444;
    color: #dddddd;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 20px;
    font-weight: bold;
    min-width: 100px;
}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton:pressed { background-color: #333333; }
QPushButton:disabled { background-color: #333333; color: #666666; }

QPushButton#start { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#start:hover { background-color: #3366bb; }
QPushButton#start:disabled { background-color: #1a1a2e; color: #555555; }

QTextEdit {
    background-color: #1e1e1e;
    color: #aaaaaa;
    border: 1px solid #444444;
    border-radius: 4px;
    font-family: 'SF Mono', 'Menlo', 'Monaco', monospace;
    font-size: 9pt;
    padding: 5px;
}

QFrame#separator {
    background-color: #444444;
    min-height: 1px;
    max-height: 1px;
}
"""


# ----------------------------------------------------------------------
# DISK‑USAGE MONITOR THREAD
# ----------------------------------------------------------------------
class DiskUsageThread(QThread):
    """Background thread that logs disk free/total space every N seconds."""
    def __init__(self, log_file: Path, workdir: Path, interval_sec: int = 5, parent=None):
        super().__init__(parent)
        self.log_file = log_file
        self.workdir = workdir
        self.interval_sec = interval_sec
        self._running = True

    def run(self):
        with open(self.log_file, "a", encoding="utf-8") as f:
            while self._running:
                try:
                    total, used, free = shutil.disk_usage(self.log_file.parent)
                    ts = datetime.now().isoformat()
                    dir_size = sum(f.stat().st_size for f in self.workdir.rglob("*") if f.is_file())
                    f.write(f"{ts},{free},{total},{dir_size}\n")
                except Exception as e:
                    f.write(f"{datetime.now().isoformat()},ERROR,{e}\n")
                time.sleep(self.interval_sec)

    def stop(self):
        self._running = False

# ----------------------------------------------------------------------
# PROCESSING THREAD
# ----------------------------------------------------------------------
class ProcessingThread(QThread):
    """Background thread for preprocessing"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str)

    def __init__(self, siril: Any, workdir: str,
                 settings: Dict[str, Any], folder_structure: str):
        super().__init__()
        self.siril = siril
        self.workdir = workdir
        self.settings = settings
        self.folder_structure = folder_structure  # 'native' or 'organized'
        self.log_area: Optional[QTextEdit] = None
        self.console_messages: List[str] = []

    # ------------------------------------------------------------------
    # Helper: robust Siril command wrapper
    # ------------------------------------------------------------------
    def _run(self, *cmd: str) -> bool:
        try:
            self.siril.cmd(*cmd)
            return True
        except RuntimeError as e:          # Siril may raise this on a crash
            self._log(f"RuntimeError in '{' '.join(cmd)}': {e}", LogColor.RED)
            return False
        except Exception as e:
            self._log(f"Error in '{' '.join(cmd)}': {e}", LogColor.RED)
            return False

    # ------------------------------------------------------------------
    # Helper: check if a sequence exists in Siril
    # ------------------------------------------------------------------
    def _sequence_exists(self, seq_name: str) -> bool:
        try:
            seqs = self.siril.get_sequence_names()
            return seq_name in seqs
        except Exception as e:
            self._log(f"Could not list sequences: {e}", LogColor.RED)
            return False

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        """Log message to both GUI and Siril with optional color formatting."""
        if self.log_area:
            cursor = self.log_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.log_area.setTextCursor(cursor)

            if color == LogColor.RED:
                self.log_area.setTextColor(Qt.GlobalColor.red)
            elif color == LogColor.GREEN:
                self.log_area.setTextColor(Qt.GlobalColor.darkGreen)
            elif color == LogColor.BLUE:
                self.log_area.setTextColor(Qt.GlobalColor.cyan)
            elif color == LogColor.SALMON:
                self.log_area.setTextColor(Qt.GlobalColor.magenta)
            else:
                self.log_area.setTextColor(Qt.GlobalColor.lightGray)

            self.log_area.append(msg)
            self.log_area.setTextColor(Qt.GlobalColor.lightGray)

        try:
            if color is not None:
                self.siril.log(msg, color=color)
            else:
                self.siril.log(msg)
        except Exception as e:
            self._log(f"Error logging to Siril: {e}", LogColor.RED)

        # Store for export
        self.console_messages.append(msg)

    # ------------------------------------------------------------------
    # Helper: create light‑xxx sub‑folders and move TIFFs into them.
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_chunk(src: Path, dest_root: Path, batch_size: int) -> None:
        """Create light‑xxx sub‑folders and move TIFFs into them."""
        if not src.is_dir():
            raise ValueError(f"'{src}' is not a directory")

        pattern = re.compile(r".*?(\d+)\.(fits?|fit)$", re.IGNORECASE)
        files: List[Path] = []

        for entry in src.iterdir():
            if not entry.is_file():
                continue
            match = pattern.fullmatch(entry.name)
            if match:
                files.append(entry)

        # Sort by the extracted frame number
        files.sort(key=lambda p: int(pattern.fullmatch(p.name).group(1)))

        def _chunk_paths(paths: List[Path], size: int):
            it = iter(paths)
            while True:
                chunk: List[Path] = []
                for _ in range(size):
                    try:
                        chunk.append(next(it))
                    except StopIteration:
                        break
                if not chunk:
                    break
                yield chunk

        chunks = list(_chunk_paths(files, batch_size))

        for idx, batch in enumerate(chunks, start=1):
            subdir_name = f"light-{idx:03d}"
            lights_dir = dest_root / subdir_name / "lights"
            try:
                lights_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeError(f"Could not create {lights_dir}: {e}")

            for f in batch:
                try:
                    shutil.move(str(f), str(lights_dir / f.name))
                except Exception as e:
                    raise RuntimeError(f"Could not move {f} to {lights_dir}: {e}")

            if idx == len(chunks) and len(batch) == 1:
                single_file = batch[0]
                dup_name = f"{single_file.stem}_dup{single_file.suffix}"
                src_path = lights_dir / single_file.name
                try:
                    shutil.copy2(src_path, lights_dir / dup_name)
                except Exception as e:
                    raise RuntimeError(f"Could not duplicate {single_file}: {e}")

    # ------------------------------------------------------------------
    def run(self):
        try:
            self._process()
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    # ------------------------------------------------------------------
    def _process(self) -> None:
        """Main processing workflow with native structure support"""
        sky_preset = SKY_PRESETS[self.settings["sky_quality"]]
        stack_method = STACKING_METHODS[self.settings["stacking_method"]]
        self.current_stack_method = stack_method   # store for final stack

        sigma_low = sky_preset["sigma_low"]
        sigma_high = sky_preset["sigma_high"]

        process_dir = self._get_process_dir(self.workdir)
        masters_dir = self._get_masters_dir(self.workdir)

        # Determine darks/lights based on folder structure
        if self.folder_structure == 'native':
            dark_file = os.path.join(self.workdir, "img-0001-dark.fits")
            lights_dir = os.path.join(self.workdir, "01-images-initial")
        else:
            darks_dir = os.path.join(self.workdir, "darks")
            lights_dir = os.path.join(self.workdir, "lights")

        # Verify folders
        if self.folder_structure == 'native':
            if not os.path.exists(dark_file):
                self.finished.emit(False, f"Dark file not found: {dark_file}")
                return
            if not os.path.exists(lights_dir):
                self.finished.emit(False, f"Lights folder not found: {lights_dir}")
                return
        else:
            if not os.path.exists(darks_dir):
                self.finished.emit(False, f"Dark folder not found: {darks_dir}")
                return
            if not os.path.exists(lights_dir):
                self.finished.emit(False, f"Light folder not found: {lights_dir}")
                return

        # Create working folders
        for d in (process_dir, masters_dir):
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                self._log(f"Could not create {d}: {e}", LogColor.RED)

        moved_count = self._move_tiff_to_reference(lights_dir)
        if moved_count:
            self._log(f"Moved {moved_count} TIFF reference image(s) to 'reference/'",
                      LogColor.SALMON)

        # Count files
        if self.folder_structure == 'native':
            num_darks = 1
            num_lights = len([f for f in glob.glob(os.path.join(lights_dir, "*.fits")) +
                              glob.glob(os.path.join(lights_dir, "*.fit"))
                              if '-dark' not in f.lower()])
        else:
            num_darks = self._count_fits(darks_dir)
            num_lights = self._count_fits(lights_dir)

        # Log configuration
        self._log(f"Sky Quality: {self.settings['sky_quality']}", LogColor.BLUE)
        self._log(f"Stacking: {self.settings['stacking_method']}", LogColor.BLUE)
        self._log(f"Structure: {self.folder_structure}", LogColor.BLUE)
        self._log(f"Found {num_darks} dark(s), {num_lights} light(s)", LogColor.BLUE)

        if num_darks == 0:
            self.finished.emit(False, "No dark frames found")
            return
        if num_lights == 0:
            self.finished.emit(False, "No light frames found")
            return

        # === CLEANUP ===
        self.progress.emit(ProcessingProgress.CLEANUP, "Cleaning previous files...")
        if self.settings.get("clean_temp", False):
            deleted = self._cleanup_folder(process_dir)
            deleted += self._cleanup_folder(masters_dir)

        # === DARK PROCESSING ===
        self.progress.emit(ProcessingProgress.DARK_PROCESSING, "Processing darks...")

        if self.folder_structure == 'native':
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
                self.siril.cmd("stack", "dark", "rej",
                                 str(sigma_low), str(sigma_high),
                                 "-nonorm", "-out=dark_stacked")

        # === LIGHT PROCESSING ===
        self.progress.emit(ProcessingProgress.LIGHT_CONVERSION, "Converting lights...")

        if self.folder_structure == 'native':
            self.siril.cmd("cd", "01-images-initial")
        else:
            self.siril.cmd("cd", "../lights")

        self.siril.cmd("convert", "light", "-out=../process")
        self.siril.cmd("cd", "../process")

        # Store sequence name for calibration step
        self.light_seq_name = "light"

        # --- Batch handling --------------------------------------------
        if self.settings.get("batch_enabled", False):
            self._process_batch_sessions(stack_method, sigma_low, sigma_high)
        else:
            # Existing single-batch workflow
            self._process_standard(stack_method, sigma_low, sigma_high)

        # === CLEANUP (post‑processing) ---------------------------------
        if self.settings.get("clean_temp", False):
            self.progress.emit(98, "Cleaning up...")
            deleted = self._cleanup_folder(process_dir)
            deleted += self._cleanup_folder(masters_dir)
            self._log(f"Cleaned {deleted} temp files", LogColor.BLUE)

        # Finalization
        self.progress.emit(ProcessingProgress.COMPLETE, "Complete!")
        self.finished.emit(True, "Processing complete!")

    # ------------------------------------------------------------------
    def _get_process_dir(self, workdir: str) -> str:
        return os.path.normpath(os.path.join(workdir, "process"))

    def _get_masters_dir(self, workdir: str) -> str:
        return os.path.normpath(os.path.join(workdir, "masters"))

    # ------------------------------------------------------------------
    def _calibrate(self, seq_name: str, stack_method: dict) -> None:
        master_dark = "../../../masters/dark_stacked" if self.settings.get("batch_enabled", False) \
                      else "../masters/dark_stacked"

        cmd = ["calibrate", seq_name,
               f"-dark={master_dark}",
               "-cc=dark", "-cfa"]

        if stack_method.get("use_drizzle"):
            cmd.append("-equalize_cfa")
        else:
            cmd.extend(["-debayer", "-equalize_cfa"])

        if not self._run(*cmd):
            raise RuntimeError("Calibration failed")

    # ------------------------------------------------------------------
    def _register(self, seq_name: str, stack_method: dict) -> None:
        use_drizzle = stack_method.get("use_drizzle", False)
        drizzle_scale = stack_method.get("drizzle_scale", 1.0)

        cmd = ["register", f"pp_{seq_name}"]

        if use_drizzle:
            cmd += [
                "-drizzle",
                f"-scale={drizzle_scale}",
                f"-pixfrac={stack_method.get('drizzle_pixfrac', 1.0)}",
                f"-kernel={stack_method.get('drizzle_kernel', 'square')}",
                f"-interp={stack_method.get('interp', 'area')}"
            ]

        if self.settings.get("two_pass", False) and (not use_drizzle or drizzle_scale == 1.0):
            cmd.append("-2pass")

        if not self._run(*cmd):
            raise RuntimeError("Registration failed")
        
        if self.settings.get("two_pass", False):
            if not use_drizzle:
                if not self._run("seqapplyreg", f"pp_{seq_name}", "-framing=max"):
                    raise RuntimeError("2pass Registration failed")
            else:
                if drizzle_scale == 1.0:
                    if not self._run(
                        "seqapplyreg", f"pp_{seq_name}",
                        "-drizzle", "-filter-round=2.5k",
                        f"-scale={drizzle_scale}",
                        f"-pixfrac={stack_method.get('drizzle_pixfrac', 1.0)}",
                        f"-kernel={stack_method.get('drizzle_kernel', 'square')}",
                        ):
                        raise RuntimeError("2pass Registration failed")
                else:
                    pass

    # ------------------------------------------------------------------
    def _stack(self, seq_name: str, stack_method: dict,
               sigma_low: float, sigma_high: float, output_name: str) -> None:
        stack_cmd = [
            "stack", f"r_pp_{seq_name}",
            "rej", str(sigma_low), str(sigma_high),
            "-norm=addscale", "-output_norm",
            "-rgb_equal", "-weight=wfwhm"
        ]
        if self.settings.get("feather_enabled", False) and self.settings.get("feather_px", 0) > 0:
            stack_cmd.append(f"-feather={self.settings['feather_px']}")

        stack_cmd.append(f"-out={output_name}")
        if not self._run(*stack_cmd):
            raise RuntimeError("Stacking failed")

    # ------------------------------------------------------------------
    def _process_standard(self,
                          stack_method: dict,
                          sigma_low: float,
                          sigma_high: float,
                          chunk_idx: Optional[int] = None,
                          total_chunks: Optional[int] = None) -> None:
        # ---- helper to compute progress ------------------------------------
        def emit(stage_rel: float, msg: str) -> None:
            """
            stage_rel : 0.0 – 58.0 (relative to the chunk)
            """
            if chunk_idx is not None and total_chunks:
                chunk_span = 58.0 / total_chunks
                start_of_chunk = 30 + (chunk_idx - 1) * chunk_span
                percent = int(start_of_chunk + (stage_rel / 58.0) * chunk_span)
            else:
                percent = int(stage_rel)          # normal mode
            self.progress.emit(percent, msg)

        # ---- calibration ------------------------------------------------------
        cal_msg = "Calibrating..."
        if chunk_idx is not None:
            cal_msg = f"Calibrating chunk {chunk_idx} of {total_chunks}"
        emit(0, cal_msg)
        try:
            self._calibrate(self.light_seq_name, stack_method)
        except Exception as e:
            self._log(f"Calibration failed{f' for chunk {chunk_idx}' if chunk_idx is not None else ''}: skipping. Error: {e}\n",
                      LogColor.SALMON)
            return

        # ---- registration ----------------------------------------------------
        reg_msg = "Registering..."
        if chunk_idx is not None:
            reg_msg = f"Registering chunk {chunk_idx} of {total_chunks}"
        emit(20, reg_msg)
        try:
            self._register(self.light_seq_name, stack_method)
        except Exception as e:
            self._log(f"Registration failed{f' for chunk {chunk_idx}' if chunk_idx is not None else ''}: skipping. Error: {e}\n",
                      LogColor.SALMON)
            return

        # ---- stacking --------------------------------------------------------
        stk_msg = "Stacking..."
        if chunk_idx is not None:
            stk_msg = f"Stacking chunk {chunk_idx} of {total_chunks}"
        emit(45, stk_msg)
        try:
            self._stack(self.light_seq_name, stack_method, sigma_low, sigma_high, "result")
        except Exception as e:
            self._log(f"Stacking failed{f' for chunk {chunk_idx}' if chunk_idx is not None else ''}: skipping. Error: {e}\n",
                      LogColor.SALMON)
            return

        # ---- finalization ----------------------------------------------------
        fin_msg = "Finalizing..."
        if chunk_idx is not None:
            fin_msg = f"Finalizing chunk {chunk_idx} of {total_chunks}"
        emit(58, fin_msg)

        try:
            self.siril.cmd("load", "result")
            self.siril.cmd("icc_remove")

            try:
                exposure_time = self.siril.get_image_fits_header("LIVETIME", default="XX")
                self.final_filename = f"result_{exposure_time}s.fit"
            except:
                self.final_filename = "result_XXXXs.fit"

            self.siril.cmd("save", "../result_$LIVETIME:%d$s")
            self.siril.cmd("cd", "..")
              
        except Exception as e:
            # If the finalization fails, log but continue
            self._log(f"Finalization failed{f' for chunk {chunk_idx}' if chunk_idx is not None else ''}: skipping. Error: {e}\n",
                      LogColor.SALMON)

    # ------------------------------------------------------------------
    def _process_batch_sessions(self,
                                stack_method: dict,
                                sigma_low: float,
                                sigma_high: float) -> None:
        batch_size = int(self.settings.get("batch_size", 100))

        # Temporarily disable feathering and 2‑pass for intermediate files
        orig_two_pass = self.settings.get("two_pass", False)
        orig_feather  = self.settings.get("feather_enabled", False)
        self.settings["two_pass"]     = False
        self.settings["feather_enabled"] = False

        # Determine intermediate stack method: if 2x upscale, fallback to recommended
        intermediate_method = stack_method
        if stack_method.get("use_drizzle") and stack_method.get("drizzle_scale", 1.0) > 1.0:
            intermediate_method = STACKING_METHODS["Bayer Drizzle (Recommended)"]

        try:
            self._prepare_chunk(Path(self.workdir, "process"),
                                Path(self.workdir, "Temp"),
                                batch_size)
        except Exception as e:
            self._log(f"Could not prepare chunks: {e}", LogColor.RED)
            return

        session_dirs = sorted(Path(self.workdir, "Temp").glob("light-*"))
        total_chunks = len(session_dirs)

        for idx, sess in enumerate(session_dirs, start=1):
            try:
                if not self._run("cd", f'"{sess}/lights"'):
                    continue
                if not self._run("convert", "light", "-out=../process"):
                    continue

                if not self._run("cd", f'"{sess}/process"'):
                    continue
                self.light_seq_name = "light"
                try:
                    self._process_standard(intermediate_method,
                                           sigma_low, sigma_high,
                                           chunk_idx=idx,
                                           total_chunks=total_chunks)
                except Exception as e:
                    self._log(f"Chunk {idx} failed: {e}\n",
                              LogColor.SALMON)
                    continue

                src_result = sess / "process" / "result.fit"
                final_stack_dir = self._create_final_stack_dir()
                if src_result.is_file():
                    dst_name = f"pp_session_{idx:03d}.fits"
                    try:
                        shutil.move(str(src_result), final_stack_dir / dst_name)
                    except Exception as e:
                        self._log(f"Could not move {src_result} to final stack: {e}",
                                  LogColor.RED)
                else:
                    self._log(f"No result file for {sess.name}",
                              LogColor.SALMON)

            except Exception as e:
                self._log(f"Processing chunk {idx}: failed. Skipping. Error: {e}\n",
                          LogColor.SALMON)
                continue

            # Clean up the temporary process folder
            if not self._cleanup_folder(sess / "process"):
                self._log(f"Could not clean temporary folder {sess}/process",
                          LogColor.RED)

        # Restore original flags before final stack
        self.settings["two_pass"] = orig_two_pass
        self.settings["feather_enabled"] = orig_feather

        # Final stack of all sessions
        self.progress.emit(ProcessingProgress.FINALIZATION, "Finalizing...")
        try:
            self._stack_final(stack_method=stack_method)
        except Exception as e:
            self._log(f"Final stack failed: {e}\n", LogColor.RED)

        temp_dir = Path(self.workdir, "Temp")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                self._log(f"Removed temporary folder: {temp_dir}",
                          LogColor.BLUE)
            except Exception as e:
                self._log(f"Could not delete Temp folder: {e}",
                          LogColor.RED)

    # ------------------------------------------------------------------
    def _create_final_stack_dir(self) -> Path:
        """Create (or re‑use) the directory that will hold all batch result FITS."""
        final_dir = Path(self.workdir, "final_stack")
        try:
            final_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Could not create final stack dir {final_dir}: {e}")
        return final_dir

    # ------------------------------------------------------------------
    def _stack_final(self, stack_method: dict) -> None:
        """Combine all batch result FITS into a single final image."""
        use_drizzle = stack_method.get("use_drizzle", False)
        drizzle_scale = stack_method.get("drizzle_scale", 1.0)

        temp_dir = Path(self.workdir) / "final_stack"
        self.siril.cmd("cd", f'"{temp_dir.as_posix()}"')
        self.siril.cmd("convert", "pp_final", f'"-out=./process"')
        self.siril.cmd("cd", "process")

        # Register the final sequence using the same method as original stack
        cmd = ["register", f"pp_final"]

        if self.settings.get("two_pass", False):
            cmd.append("-2pass")

        if not self._run(*cmd):
            raise RuntimeError("Registration failed")
        
        if self.settings.get("two_pass", False):
                if not self._run(
                        "seqapplyreg", f"pp_final",
                        "-filter-round=2.5k",
                        f"-scale={drizzle_scale}",
                        ):
                    raise RuntimeError("2pass Registration failed")
               
        stack_cmd = [
            "stack", "r_pp_final",
            "rej", 
            "3",
            "3",
            "-norm=addscale", 
            "-output_norm",
            "-weight=wfwhm"
        ]

        if self.settings.get("feather_enabled", False) and self.settings.get("feather_px", 0) > 0:
            stack_cmd.append(f"-feather={self.settings['feather_px']}")

        if not self._run(*stack_cmd):
            raise RuntimeError("Stacking failed")
        
        num_sessions = len(list(temp_dir.glob("pp_session_*.fits")))

        final_name = f"final_stacked_batch{self.settings['batch_size']}_sessions{num_sessions}.fits"
        final_out = Path(self.workdir) / final_name
        
        self.siril.cmd("load", "r_pp_final_stacked")
        self.siril.cmd("save", f"../../{final_name}")
        self.siril.cmd("cd", "../..")
        self.siril.log(f"Final stacked image: {final_out}")
        self.siril.cmd("load", f'"{final_name}"')

    # ------------------------------------------------------------------
    def _count_fits(self, folder: str) -> int:
        count = 0
        for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS']:
            count += len(glob.glob(os.path.join(folder, ext)))
        return count

    # ------------------------------------------------------------------
    def _move_tiff_to_reference(self, lights_dir: str) -> int:
        tiff_files = []
        for pattern in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tiff_files.extend(glob.glob(os.path.join(lights_dir, pattern)))

        if not tiff_files:
            return 0

        reference_dir = os.path.join(self.workdir, "reference")
        try:
            os.makedirs(reference_dir, exist_ok=True)
        except Exception as e:
            self._log(f"Could not create reference dir: {e}", LogColor.RED)
            return 0

        moved_count = 0
        for tiff_file in tiff_files:
            try:
                filename = os.path.basename(tiff_file)
                dest_path = os.path.join(reference_dir, filename)
                shutil.move(tiff_file, dest_path)
                self._log(f"Moved reference image: {filename} → reference/",
                          LogColor.SALMON)
                moved_count += 1
            except Exception as e:
                self._log(f"Warning: Could not move {tiff_file}: {e}",
                          LogColor.SALMON)

        return moved_count

    # ------------------------------------------------------------------
    def _cleanup_folder(self, folder: str) -> int:
        if not os.path.exists(folder):
            return 0

        count = 0
        try:
            for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS',
                        '*.seq', '*conversion.txt']:
                for f in glob.glob(os.path.join(folder, ext)):
                    try:
                        os.remove(f)
                        count += 1
                    except OSError as e:
                        self._log(f"Warning: Could not remove {f}: {e}",
                                  LogColor.SALMON)
                        continue

            for subdir in ['cache', 'drizztmp', 'other']:
                subdir_path = os.path.join(folder, subdir)
                if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                    try:
                        shutil.rmtree(subdir_path)
                        count += 1
                    except OSError as e:
                        self._log(f"Warning: Could not remove {subdir_path}: {e}",
                                  LogColor.SALMON)
                        continue
            
            final_stack_dir = Path(folder).parent / "final_stack"
            if final_stack_dir.exists() and final_stack_dir.is_dir():
                try:
                    shutil.rmtree(final_stack_dir)
                    count += 1
                    self._log(f"Removed final stack directory: {final_stack_dir}",
                              LogColor.BLUE)
                except Exception as e:
                    self._log(f"Could not delete final_stack: {e}", LogColor.RED)

        except Exception as e:
            self._log(f"Error during cleanup: {e}", LogColor.RED)

        return count

# ----------------------------------------------------------------------
# MAIN GUI
# ----------------------------------------------------------------------
class VesperaProGUI(QDialog):
    """Full‑featured Vespera preprocessing dialog"""

    def __init__(self, siril: Any, app: QApplication):
        super().__init__()
        self.siril = siril
        self.app = app
        self.worker: Optional[ProcessingThread] = None
        self.disk_thread: Optional[DiskUsageThread] = None
        self.qsettings = QSettings("Vespera", "Preprocessing")
        self.current_settings: Dict[str, Any] = {}

        self.setWindowTitle(f"Vespera — Preprocessing v{VERSION}")
        self.setMinimumWidth(550)
        self.setMinimumHeight(900)
        self.resize(550, 900)
        self.setStyleSheet(DARK_STYLESHEET)

        self.reloading = False 

        self._setup_ui()
        self._load_settings()
        self._check_folders()

    def _log(self, msg: str, color: Optional[LogColor] = None) -> None:
        """Append a coloured message to the GUI log area."""
        if self.log_area:
            cursor = self.log_area.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.log_area.setTextCursor(cursor)

            if color == LogColor.RED:
                self.log_area.setTextColor(Qt.GlobalColor.red)
            elif color == LogColor.GREEN:
                self.log_area.setTextColor(Qt.GlobalColor.darkGreen)
            elif color == LogColor.BLUE:
                self.log_area.setTextColor(Qt.GlobalColor.cyan)
            elif color == LogColor.SALMON:
                self.log_area.setTextColor(Qt.GlobalColor.magenta)
            else:
                self.log_area.setTextColor(Qt.GlobalColor.lightGray)

            self.log_area.append(msg)
            self.log_area.setTextColor(Qt.GlobalColor.lightGray)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QVBoxLayout()
        title = QLabel("")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title)

        layout.addLayout(header)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_main_tab(), "Main")
        tabs.addTab(self._create_info_tab(), "Info")
        layout.addWidget(tabs)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
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

        # Folder Status
        status_group = QGroupBox("Folder Status")
        status_layout = QVBoxLayout(status_group)

        wd_row = QHBoxLayout()
        self.lbl_workdir = QLabel("Working directory: ...")
        self.btn_browse_dir = QPushButton("Browse…")
        self.btn_browse_dir.clicked.connect(self._browse_working_directory)
        self.btn_browse_dir.setStyleSheet("""
            QPushButton {
            padding: 2px 6px;
            }
        """)
        wd_row.addWidget(self.lbl_workdir)
        wd_row.addStretch()
        wd_row.addWidget(self.btn_browse_dir)
        status_layout.addLayout(wd_row)

        folder_row = QHBoxLayout()
        self.lbl_darks = QLabel("Darks: checking...")
        self.lbl_lights = QLabel("Lights: checking...")
        folder_row.addWidget(self.lbl_darks)
        folder_row.addWidget(self.lbl_lights)
        status_layout.addLayout(folder_row)

        self.lbl_structure = QLabel("")
        self.lbl_structure.setObjectName("info")
        self.lbl_structure.setWordWrap(True)
        status_layout.addWidget(self.lbl_structure)

        layout.addWidget(status_group)

        # Sky Quality
        sky_group = QGroupBox("Sky Quality (Location)")
        sky_layout = QVBoxLayout(sky_group)

        self.combo_sky = QComboBox()
        for name in SKY_PRESETS.keys():
            self.combo_sky.addItem(name)
        self.combo_sky.currentTextChanged.connect(self._on_sky_changed)
        sky_layout.addWidget(self.combo_sky)

        self.lbl_sky_desc = QLabel("")
        self.lbl_sky_desc.setObjectName("info")
        self.lbl_sky_desc.setWordWrap(True)
        sky_layout.addWidget(self.lbl_sky_desc)

        layout.addWidget(sky_group)

        # Stacking Method
        stack_group = QGroupBox("Stacking Method")
        stack_layout = QVBoxLayout(stack_group)

        self.combo_stack = QComboBox()
        for idx, (name, config) in enumerate(STACKING_METHODS.items()):
            self.combo_stack.addItem(name)
            if "tooltip" in config:
                self.combo_stack.setItemData(idx, config["tooltip"],
                                            Qt.ItemDataRole.ToolTipRole)
        self.combo_stack.currentTextChanged.connect(self._on_stack_changed)
        stack_layout.addWidget(self.combo_stack)

        self.lbl_stack_desc = QLabel("")
        self.lbl_stack_desc.setObjectName("info")
        self.lbl_stack_desc.setWordWrap(True)
        stack_layout.addWidget(self.lbl_stack_desc)

        layout.addWidget(stack_group)

        # Stacking Options
        options_group = QGroupBox("Stacking Options")
        options_layout = QVBoxLayout(options_group)

        # Feather option
        feather_layout = QHBoxLayout()
        self.chk_feather = QCheckBox("Enable Feathering")
        self.chk_feather.setToolTip(
            "Feathering blends image edges to reduce artifacts (0‑100px)")

        self.feather_slider = QSpinBox()
        self.feather_slider.setRange(0, 50)
        self.feather_slider.setValue(0)
        self.feather_slider.setSuffix(" px")
        self.feather_slider.setToolTip(
            "Feathering distance in pixels (0 to disable)")

        feather_layout.addWidget(self.chk_feather)
        feather_layout.addWidget(self.feather_slider)
        options_layout.addLayout(feather_layout)

        hbox_two_pass = QHBoxLayout()
        self.chk_two_pass = QCheckBox("2‑Pass Registration")
        self.chk_two_pass.setToolTip(
            "Perform two-pass registration with framing for optimal alignment")
        hbox_two_pass.addWidget(self.chk_two_pass)
        hbox_two_pass.addStretch(0)
        options_layout.addLayout(hbox_two_pass) 

        hbox_batch = QHBoxLayout()
        self.chk_batch = QCheckBox("Batch Processing")
        self.chk_batch.setToolTip(
            "Split the light frames into batches before stacking (recommended for large sessions)")
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(10, 100)
        self.spin_batch_size.setValue(0)
        self.spin_batch_size.setSuffix(" images/chunk")
        self.spin_batch_size.setToolTip(
            "Number of light frames per batch")
        hbox_batch.addWidget(self.chk_batch)
        hbox_batch.addWidget(self.spin_batch_size)
        options_layout.addLayout(hbox_batch)

        hbox_clean_temp = QHBoxLayout()
        self.chk_clean_temp = QCheckBox("Clean temporary files")
        self.chk_clean_temp.setToolTip(
            "Delete process/ and masters/ folders")
        hbox_clean_temp.addWidget(self.chk_clean_temp)
        hbox_clean_temp.addStretch(0)
        options_layout.addLayout(hbox_clean_temp)

        layout.addWidget(options_group)
        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    def _create_info_tab(self) -> QWidget:
        """Information/help tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        info_text = """
        <h3 style="color:#88aaff;">Vespera Preprocessing – Quick Start</h3>

        <p><b>No Setup Required!</b> Point the plugin at your Vespera observation folder and press 
        <strong>Start Processing</strong>.</p>

        <li><b>- Folder detection</b>: The plugin auto-detects either the <code>darks/ & lights/</code> structure
            or Vespera’s native layout (<code>01‑images-initial</code>).  If no valid structure is found,
            you’ll be prompted to select a working directory.</li>
        <li><b>- Sky quality</b>: Choose your site’s Bortle rating to set sigma limits and background sampling automatically.</li>
        <li><b>- Stacking method</b>:  
            <ul style="margin-left:10px;">
            <li><code>Bayer Drizzle (Recommended)</code> – optimal for rotating fields.</li>
            <li><code>Standard Registration</code> – faster for short sessions.</li>
            </ul>
        </li>
        <li><b>- Feathering</b>: Enable to blend image edges (0–50 px).  Useful for large mosaics.</li>
        <li><b>- 2‑Pass Registration</b>: Adds framing for better alignment when field rotation is significant.</li>
        <li><b>- Batch processing</b>: Splits lights into chunks (default 20 images) before stacking; recommended for large sessions.
        <br>Slower processing but will save disk space.</li>
        <li><b>- Clean temporary files</b>: Removes the <code>process/</code> and temporary folders after completion.</li>
        <li><b>- Output</b>: A FITS image named <code>result_<i>exposure</i>s.fit</code> 
            (or <code>final_stacked_batch…fits</code> for batch mode) is saved in the working directory.</li>
        </ul>

  
        <p>
        <h4 style="color: #88aaff;">Known Limitations</h4>
        <p> Using a low number of images can lead to the script to crash. Increase the size of the chunks. Min. recommended 20</p>
        <p> 2pass registration only works well with large chunks.</p>
        
        <h4 style="color: #88aaff;">Credits</h4>
        Developed for SIRIL.<br>
        (c) G. Trainar (2026)
        """
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #cccccc; font-size: 12pt;")
        layout.addWidget(info_label)

        return widget

    # ------------------------------------------------------------------
    def _on_sky_changed(self, name: str) -> None:
        if name in SKY_PRESETS:
            self.lbl_sky_desc.setText(SKY_PRESETS[name]["description"])

    def _on_stack_changed(self, name: str) -> None:
        if name in STACKING_METHODS:
            self.lbl_stack_desc.setText(STACKING_METHODS[name]["description"])

    # ------------------------------------------------------------------
    def _load_settings(self) -> None:
        """Load saved settings"""
        self.combo_sky.setCurrentText(
            self.qsettings.value("sky_quality", "Bortle 3-4 (Rural)"))
        self.combo_stack.setCurrentText(
            self.qsettings.value("stacking_method",
                                 "Bayer Drizzle (Recommended)"))
        self.chk_feather.setChecked(
            self.qsettings.value("feather_enabled", False, type=bool))
        self.feather_slider.setValue(
            self.qsettings.value("feather_px", 0, type=int))
        self.chk_two_pass.setChecked(
            self.qsettings.value("two_pass", False, type=bool))
        self.chk_clean_temp.setChecked(
            self.qsettings.value("clean_temp", False, type=bool))
        self.chk_batch.setChecked(
            self.qsettings.value("batch_enabled", False, type=bool))
        self.spin_batch_size.setValue(
            self.qsettings.value("batch_size", 100, type=int))

        # Trigger description updates
        self._on_sky_changed(self.combo_sky.currentText())
        self._on_stack_changed(self.combo_stack.currentText())

    def _save_settings(self) -> None:
        """Save current settings"""
        self.qsettings.setValue("sky_quality", self.combo_sky.currentText())
        self.qsettings.setValue("stacking_method",
                                self.combo_stack.currentText())
        self.qsettings.setValue("feather_enabled", self.chk_feather.isChecked())
        self.qsettings.setValue("feather_px", self.feather_slider.value())
        self.qsettings.setValue("two_pass", self.chk_two_pass.isChecked())
        self.qsettings.setValue("clean_temp", self.chk_clean_temp.isChecked())
        self.qsettings.setValue("batch_enabled", self.chk_batch.isChecked())
        self.qsettings.setValue("batch_size", self.spin_batch_size.value())

    # ------------------------------------------------------------------
    def _browse_working_directory(self) -> None:
        """
        Open a QFileDialog starting from the current working directory,
        change Siril’s CWD, and re‑run folder detection.
        """
        current_dir = getattr(self, "workdir", os.path.expanduser("~"))

        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            current_dir
        )
        if selected:
            try:
                self.siril.cmd("cd", f'"{selected}"')
                self.workdir = selected
            except Exception as e:
                self._log(f"Could not change Siril working dir: {e}", LogColor.RED)

            self._check_folders()

    def _check_folders(self) -> None:
        """Check folder status - supports both organized and native Vespera structure"""
        try:
            workdir = self.siril.get_siril_wd()
            self.workdir = workdir
            self.lbl_workdir.setText(f"Working directory: {self.workdir}")

            # First check for organized structure (darks/ and lights/ folders)
            darks_dir = os.path.join(workdir, "darks")
            lights_dir = os.path.join(workdir, "lights")

            num_darks_organized = self._count_fits(darks_dir) if os.path.exists(darks_dir) else 0
            num_lights_organized = self._count_fits(lights_dir) if os.path.exists(lights_dir) else 0

            # Check for native Vespera structure
            native = self._detect_native_structure(workdir)

            # Determine which structure to use
            if num_darks_organized > 0 and num_lights_organized > 0:
                self.folder_structure = 'organized'
                num_darks = num_darks_organized
                num_lights = num_lights_organized
                self.lbl_structure.setText("Using organized folders (darks/, lights/)")
                self.lbl_structure.setStyleSheet("color: #88aaff;")
            elif native:
                self.folder_structure = 'native'
                num_darks = native['num_darks']
                num_lights = native['num_lights']
                self.lbl_structure.setText("Using Vespera native structure")
                self.lbl_structure.setStyleSheet("color: #88aaff;")
            else:
                # No structure detected – ask the user to pick a folder
                self.folder_structure = None
                num_darks = 0
                num_lights = 0
                self.lbl_structure.setText("No valid folder structure detected")
                self.lbl_structure.setStyleSheet("color: #ff8888;")

            if num_darks > 0:
                self.lbl_darks.setText(f"✓ Darks: {num_darks}")
                self.lbl_darks.setStyleSheet("color: #88ff88;")
            else:
                self.lbl_darks.setText("✗ Darks: not found")
                self.lbl_darks.setStyleSheet("color: #ff8888;")

            if num_lights > 0:
                self.lbl_lights.setText(f"✓ Lights: {num_lights}")
                self.lbl_lights.setStyleSheet("color: #88ff88;")
            else:
                self.lbl_lights.setText("✗ Lights: not found")
                self.lbl_lights.setStyleSheet("color: #ff8888;")

            self.btn_start.setEnabled(num_darks > 0 and num_lights > 0)

        except Exception as e:
            self._log(f"Error: {e}")
            self.btn_start.setEnabled(False)

    def _detect_native_structure(self, workdir: str) -> Optional[Dict[str, Any]]:
        """Detect native Vespera folder structure"""
        workdir = os.path.normpath(workdir)

        dark_files = set()
        for pattern in ['*-dark.fits', '*-dark.fit',
                        '*-dark.FITS', '*-dark.FIT']:
            dark_files.update(glob.glob(os.path.join(workdir, pattern)))

        images_initial = os.path.join(workdir, "01-images-initial")
        light_files = set()
        if os.path.exists(images_initial):
            for pattern in ['*.fits', '*.fit',
                            '*.FITS', '.FIT']:
                all_fits = glob.glob(os.path.join(images_initial, pattern))
                light_files.update([f for f in all_fits if '-dark' not in f.lower()])

        dark_files = list(dark_files)
        light_files = list(light_files)

        if dark_files and light_files:
            return {
                'dark_files': dark_files,
                'light_files': light_files,
                'num_darks': len(dark_files),
                'num_lights': len(light_files),
                'images_initial': images_initial
            }
        return None

    def _count_fits(self, folder: str) -> int:
        """Return the number of FITS files in *folder*."""
        count = 0
        for ext in ['*.fit', '*.fits', '*.FIT', '*.FITS']:
            count += len(glob.glob(os.path.join(folder, ext)))
        return count

    # ------------------------------------------------------------------
    def _start_processing(self) -> None:
        """Start the processing thread and disk‑usage monitor"""
        self._save_settings()
        self.btn_start.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Processing...")
        self.log_area.clear()

        temp_dir = Path(self.siril.get_siril_wd()) / "Temp"
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)          # delete the whole Temp folder
                self._log("Previous Temp directory cleaned before chunk calculation", LogColor.BLUE)
            except Exception as e:
                self._log(f"Could not delete Temp folder: {e}", LogColor.RED)

        settings = {
            "sky_quality": self.combo_sky.currentText(),
            "stacking_method": self.combo_stack.currentText(),
            "feather_px": self.feather_slider.value(),
            "feather_enabled": self.chk_feather.isChecked(), 
            "two_pass": self.chk_two_pass.isChecked(),
            "clean_temp": self.chk_clean_temp.isChecked(),
            "batch_enabled": self.chk_batch.isChecked(),
            "batch_size": self.spin_batch_size.value()
        }
        self.current_settings = settings

        try:
            workdir = self.siril.get_siril_wd()

            if self.folder_structure == 'organized':
                lights_dir = os.path.join(workdir, "lights")
            else:   # native Vespera layout
                lights_dir = os.path.join(workdir, "01-images-initial")

            num_lights = self._count_fits(lights_dir)
            if settings["batch_enabled"]:
                from math import ceil
                self.num_chunks = ceil(num_lights / settings["batch_size"])
            else:
                self.num_chunks = 1

            # --- start processing thread ------------------------------------
            self.worker = ProcessingThread(self.siril, workdir,
                                          settings, self.folder_structure)
            self.worker.log_area = self.log_area
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.worker.log.connect(self._log)
            self.worker.start()

            # --- create disk‑usage log file with header --------------------
            logs_dir = Path(workdir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            disk_log_file = logs_dir / f"disk_usage_{timestamp}.log"

            header_line = self._config_summary()
            with open(disk_log_file, "w", encoding="utf-8") as f:
                f.write(header_line + "\n")

            # --- start disk‑usage thread ------------------------------------
            workdir_path = Path(self.siril.get_siril_wd())
            self.disk_thread = DiskUsageThread(disk_log_file, workdir=workdir_path, interval_sec=5)
            self.disk_thread.start()

        except Exception as e:
            self._log(f"Start error: {e}", LogColor.RED)
            self.btn_start.setEnabled(True)

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.setValue(percent)
        self.status.setText(message)
        self.status.setStyleSheet("color: #ffcc00;")
        self.app.processEvents()

    def _on_finished(self, success: bool, message: str) -> None:
        """Called when the processing thread finishes"""
        self.btn_start.setEnabled(True)

        # Stop disk‑usage monitoring
        if hasattr(self, 'disk_thread'):
            self.disk_thread.stop()
            self.disk_thread.wait()

        # Write console log to file
        if hasattr(self, 'worker'):
            self._write_console_log()

        # Update GUI (unchanged)
        if success:
            self.status.setText("✓ " + message)
            self.status.setStyleSheet("color: #88ff88;")
            self._log("Stacking complete!", LogColor.GREEN)
            self._log("\n", LogColor.GREEN)

            try:
                self.siril.log("Stacking Complete!", color=LogColor.GREEN)
            except:
                pass
        else:
            self.status.setText("✗ " + message)
            self.status.setStyleSheet("color: #ff8888;")
            self._log(f"FAILED: {message}", LogColor.RED)

    def _config_summary(self) -> str:
        """Return a one‑line summary of the current configuration."""
        s = self.current_settings
        stacking = s.get("stacking_method", "Unknown")
        feather = "Yes" if s.get("feather_enabled") else "No"
        two_pass = "Yes" if s.get("two_pass") else "No"
        batch = "Yes" if s.get("batch_enabled") else "No"
        chunks = getattr(self, "num_chunks", 1)
        return (f"Stacking method: {stacking}, Feathering: {feather}, "
                f"2‑Pass: {two_pass}, Batch: {batch}, Chunks: {chunks}")

    def _write_console_log(self) -> None:
        """Persist all Siril console messages to a log file."""
        if not hasattr(self, 'worker'):
            return
        logs_dir = Path(self.workdir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"siril_console_{ts}.log"

        header_line = self._config_summary()
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(header_line + "\n")
                for msg in self.worker.console_messages:
                    f.write(msg + "\n")
        except Exception as e:
            # If we cannot write the log, still report it to GUI
            self._log(f"Failed to write console log: {e}", LogColor.RED)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
_global_gui = None   # for crash handler

def main() -> None:
    """Main entry point"""
    global _global_gui
    try:
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        siril = s.SirilInterface()

        try:
            siril.connect()
        except Exception as e:
            QMessageBox.critical(None, "Connection Error",
                                 f"Could not connect to Siril.\n{e}")
            return

        gui = VesperaProGUI(siril, app)
        _global_gui = gui  # keep a global reference for crash handler

        def _crash_handler(exc_type, exc_value, tb):
            """Flush logs if the interpreter crashes."""
            gui = _global_gui
            if gui:
                disk_thread = getattr(gui, 'disk_thread', None)
                if disk_thread:
                    try:
                        disk_thread.stop()
                        disk_thread.wait()
                    except Exception as e:
                        gui._log(f"Error stopping disk thread: {e}", LogColor.RED)
                if hasattr(gui, 'worker'):
                    gui._write_console_log()
            # Re‑raise the original exception to show it
            sys.__excepthook__(exc_type, exc_value, tb)

        sys.excepthook = _crash_handler

        gui.show()
        app.exec()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
