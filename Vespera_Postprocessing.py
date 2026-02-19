##############################################
# Vespera Postprocessing
# One‑Click Image Preparation Pipeline
# For Vespera Smart Telescope
# Author: G. Trainar
# Contact: https://github.com/gtrainar
##############################################

# MIT License
# Version 1.1.0

"""
Overview
--------
A streamlined preparation plugin for Vespera 16‑bit TIFF images that
automates the tedious pre‑stretch workflow:

1. Background Extraction (GraXpert AI or Siril RBF)
2. Plate Solving (for coordinate metadata)
3. Spectrophotometric Color Correction (SPCC)
4. Denoising (multiple engine choices)
5. Auto or VeraLux HMS for stretching

This plugin bridges the gap between Vespera's output and the final stretch,
eliminating repetitive manual steps while preserving full control over each stage.
"""

import sys
import os
import threading

try:
    import sirilpy as s
    from sirilpy import SirilInterface, SirilError, LogColor
except ImportError:
    print("Error: sirilpy module not found. This script must be run within Siril.")
    sys.exit(1)

s.ensure_installed("PyQt6", "numpy")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QGroupBox, QRadioButton, QButtonGroup,
    QCheckBox, QSlider, QProgressBar, QMessageBox, QFrame,
    QLineEdit, QInputDialog, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QProcess
from PyQt6.QtGui import QFont

VERSION = "1.0.1"

# ---------------------
#  CHANGE LOG
# ---------------------
CHANGELOG = """
Version 1.1.0 (2026-02)
- Autostretch
- GUI refresh

Version 1.0.0 (2026-01)
- Advanced plate solving with SIMBAD integration
- Manual DSO name entry fallback
- SPCC
- Vespera II and Vespera Pro support
"""

# ---------------------
#  DARK THEME
# ---------------------
DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 12pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox {
    border: 1px solid #444444;
    margin-top: 10px;
    font-weight: bold;
    border-radius: 4px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5x;
    color: #88aaff;
}
QLabel { color: #cccccc; }
QRadioButton, QCheckBox { color: #cccccc; spacing: 5px; }
QRadioButton::indicator, QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #666666;
    background: #3c3c3c;
    border-radius: 7px;
}
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius: 0.4,
        fx:0.5, fy:0.5, stop:0 #ffffff, stop:1 #285299);
    border: 1px solid #88aaff;
}
QCheckBox::indicator:checked {
    background-color: #285299;
    border: 1px solid #88aaff;
}
QSlider::groove:horizontal {
    background: #444444;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #aaaaaa;
    width: 14px; height: 14px;
    margin: -4px 0;
    border-radius: 7px;
    border: 1px solid #555555;
}
QSlider::handle:horizontal:hover { background-color: #ffffff; }
QPushButton {
    background-color: #444444;
    color: #dddddd;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton {
    background-color: #285299;
    border: 1px solid #1e3f7a;
    font-size: 12pt;
    padding: 12px;
}
QPushButton#PrepButton:hover { background-color: #355ea1; }
QProgressBar {
    border: 1px solid #555555;
    border-radius: 3px;
    text-align: center;
    background-color: #333333;
}
QProgressBar::chunk { background-color: #285299; }
QFrame#Separator { background-color: #444444; }
"""


class VesperaPlateSolver:
    """Advanced plate solving for Vespera 16‑bit TIFF images that
    bridge the gap between Vespera's output and the final stretch,
    eliminating repetitive manual steps while preserving full control over each stage.
    """

    def __init__(self, siril_interface, filename=None, pixel_size_um=2.00):
        self.siril = siril_interface
        self.filename = filename
        self.dso_name = None
        self.applied_coordinates = None
        self.focal_length_mm = 249.47
        self.pixel_size_um = pixel_size_um

        # Extract DSO name immediately
        if filename:
            self._extract_dso_name()

    def _extract_dso_name(self):
        """Extract DSO name from filename."""
        try:
            import os

            filename = os.path.basename(str(self.filename))
            filename_without_ext = os.path.splitext(filename)[0]

            # Split on first underscore or dash
            if '_' in filename_without_ext:
                dso_name = filename_without_ext.split('_', 1)[0].strip()
            elif '-' in filename_without_ext:
                dso_name = filename_without_ext.split('-', 1)[0].strip()
            else:
                dso_name = filename_without_ext.strip()

            # Validate the extracted DSO name
            if self._validate_dso_name(dso_name):
                self.dso_name = dso_name
            else:
                self.dso_name = None

        except Exception as e:
            self.siril.log(f"DSO extraction error: {e}", LogColor.SALMON)

    def _validate_dso_name(self, dso_name):
        """Validate extracted DSO name."""
        if not dso_name or len(dso_name.strip()) == 0:
            self.siril.log(f"Validation failed: DSO name is empty or None", LogColor.SALMON)
            return False

        # Check if DSO name contains valid characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -')
        if not all(c in valid_chars for c in dso_name):
            invalid_chars = [c for c in dso_name if c not in valid_chars]
            self.siril.log(f"Validation failed: Invalid characters '{invalid_chars}' in '{dso_name}'", LogColor.SALMON)
            return False

        return True

    def plate_solve(self):
        """Execute plate solving with accuracy priority."""
        try:
            command_parts = ["platesolve"]

            if self.applied_coordinates:
                ra, dec = self.applied_coordinates
                formatted_coords = self._format_coordinates_for_platesolve(
                    ra, dec
                )
                command_parts.append(formatted_coords)

            command_parts.extend([
                f"-focal={self.focal_length_mm}",
                f"-pixelsize={self.pixel_size_um}"
            ])

            full_command = " ".join(command_parts)
            self.siril.cmd(full_command)

            return True

        except Exception as e:
            self.siril.log(f"Plate solve error: {e}", LogColor.SALMON)
            return False

    def _format_coordinates_for_platesolve(self, ra, dec):
        """Convert RA/DEC from SIMBAD format to platesolve format."""
        try:
            ra_clean = ra.replace('h', ':').replace('m', ':').replace('s', '')
            if ra_clean.endswith(':'):
                ra_clean = ra_clean[:-1]

            dec_clean = dec.replace('d', ':').replace('m', ':').replace('s', '')
            if dec_clean.endswith(':'):
                dec_clean = dec_clean[:-1]

            ra_clean = self._ensure_ra_format(ra_clean)
            dec_clean = self._ensure_dec_format(dec_clean)

            return f'"{ra_clean}, {dec_clean}"'

        except Exception as e:
            self.siril.log(f"Coordinate formatting error: {e}", LogColor.SALMON)
            return f'"{ra}, {dec}"'

    def _ensure_ra_format(self, ra):
        """Ensure RA is in HH:MM:SS format with proper padding."""
        parts = ra.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return f"{h.zfill(2)}:{m.zfill(2)}:{s.zfill(2)}"
        elif len(parts) == 2:
            h, m = parts
            return f"{h.zfill(2)}:{m.zfill(2)}:00"
        else:
            return ra

    def _ensure_dec_format(self, dec):
        """Ensure DEC is in ±DD:MM:SS format with proper padding and sign."""
        if dec.startswith('-') or dec.startswith('+'):
            sign = dec[0]
            parts = dec[1:].split(':')
        else:
            sign = '+'
            parts = dec.split(':')

        if len(parts) == 3:
            d, m, s = parts
            return f"{sign}{d.zfill(2)}:{m.zfill(2)}:{s.zfill(2)}"
        elif len(parts) == 2:
            d, m = parts
            return f"{sign}{d.zfill(2)}:{m.zfill(2)}:00"
        else:
            return dec if dec.startswith(('+', '-')) else '+' + dec

    def _query_simbad_coordinates(self, dso_name):
        """Query SIMBAD database using proper URL API to get RA/DEC coordinates."""
        import urllib.parse
        import urllib.request

        try:
            base_url = "https://simbad.cds.unistra.fr/simbad/sim-id"

            params = {
                'output.format': 'ASCII',
                'Ident': dso_name
            }

            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read().decode('utf-8')

                ra = None
                dec = None

                for line in data.split('\n'):
                    if line.startswith('RA(J2000)'):
                        parts = line.split()
                        if len(parts) >= 2:
                            ra = parts[1]
                    elif line.startswith('DE(J2000)'):
                        parts = line.split()
                        if len(parts) >= 2:
                            dec = parts[1]
                    elif 'Coordinates(' in line and ':' in line:
                        coord_part = line.split(':', 1)[1].strip()
                        coord_parts = coord_part.split()
                        if len(coord_parts) >= 3:
                            ra_h, ra_m, ra_s = coord_parts[0], coord_parts[1], coord_parts[2]
                            ra = f"{ra_h}h{ra_m}m{ra_s}s"

                            dec_d, dec_m = coord_parts[3], coord_parts[4]
                            dec_s = coord_parts[5] if len(coord_parts) > 5 else "0"
                            dec = f"{dec_d}d{dec_m}m{dec_s}s"

                if ra and dec:
                    self.siril.log(f"SIMBAD coordinates found: RA={ra}, DEC={dec}", LogColor.GREEN)
                    return ra, dec
                else:
                    self.siril.log("No coordinates found in SIMBAD response", LogColor.SALMON)
                    return None

        except Exception as e:
            self.siril.log(f"SIMBAD query error: {e}", LogColor.SALMON)
            return None


class PrepWorker(QThread):
    """Background thread for running the preparation pipeline."""
    progress = pyqtSignal(int, str)  # percent, status message
    finished = pyqtSignal(bool, str)  # success, message
    manual_dso_request = pyqtSignal()  # Request manual DSO entry
    update_dso_input_visibility = pyqtSignal()  # Update DSO input visibility

    def __init__(self, siril, options, dso_name=None):
        super().__init__()
        self.siril = siril
        self.options = options
        self.manual_dso_name = dso_name
        self.manual_dso_event = threading.Event()
        self.provided_dso_name = None

    def run(self):
        try:
            total_steps = self._count_steps()
            current_step = 0

            # Step 1: Background Extraction
            if self.options['bge_method'] != 'none':
                current_step += 1
                pct = int(current_step / total_steps * 100)
                self.progress.emit(pct, "Extracting background...")
                self._run_background_extraction()

            # Step 2: Plate Solve
            plate_solve_success = False
            if self.options['plate_solve']:
                current_step += 1
                pct = int(current_step / total_steps * 100)
                self.progress.emit(pct, "Plate solving...")

                plate_solve_success = self._run_plate_solve()
                if not plate_solve_success:
                    self.siril.log("Plate solving failed, continuing with other processing...", LogColor.SALMON)

            # Step 3: Spectrophotometric Color Correction (SPCC)
            if self.options['spcc'] and plate_solve_success:
                current_step += 1
                pct = int(current_step / total_steps * 100)
                self.progress.emit(pct, "Color calibrating...")
                self._run_spcc()
            elif self.options['spcc'] and not plate_solve_success:
                self.siril.log("Skipping color calibration - requires plate solved image", LogColor.SALMON)

            # Step 4: Denoise
            if self.options['denoise_method'] != 'none':
                current_step += 1
                pct = int(current_step / total_steps * 100)
                self.progress.emit(pct, f"Denoising ({self.options['denoise_method']})...")
                self._run_denoise()

            # Step 5: Auto‑stretch
            if self.options['autostretch']:
                current_step += 1
                pct = int(current_step / total_steps * 100)
                self.progress.emit(pct, "Auto‑stretching...")
                self._run_autostretch()

            self.progress.emit(100, "Complete!")
            self.finished.emit(True, "Image processed!")

        except Exception as e:
            self.finished.emit(False, str(e))

    def _count_steps(self):
        """Count total processing steps."""
        steps = 0
        if self.options['bge_method'] != 'none':
            steps += 1
        if self.options['plate_solve']:
            steps += 1
        if self.options['spcc']:
            steps += 1
        if self.options['denoise_method'] != 'none':
            steps += 1
        if self.options['autostretch']:
            steps += 1
        return max(steps, 1)

    def _run_background_extraction(self):
        """Run background extraction based on selected method."""
        method = self.options['bge_method']

        if method == 'graxpert':
            smoothing = self.options['bge_smoothing']
            # Call GraXpert-AI.py via pyscript
            self.siril.cmd("pyscript", "GraXpert-AI.py",
                          "-bge", f"-smoothing={smoothing}")
        elif method == 'siril_rbf':
            # Use Siril's built‑in RBF background extraction
            self.siril.cmd("subsky", "-rbf", "-samples=60",
                          "-tolerance=1.0", "-smooth=0.5")

    def _run_plate_solve(self):
        """Run plate solving with DSO identification and coordinate lookup."""
        try:
            filename = self._get_current_filename()
            plate_solver = VesperaPlateSolver(self.siril, filename,
                                              pixel_size_um=self.options['pixel_size_um'])

            # Use manual DSO name if provided
            if self.manual_dso_name and plate_solver._validate_dso_name(self.manual_dso_name):
                plate_solver.dso_name = self.manual_dso_name

            # Get coordinates from SIMBAD
            self._get_simbad_coordinates(plate_solver)

            if not plate_solver.applied_coordinates:
                self.siril.log("Cannot plate solve without valid coordinates", LogColor.SALMON)
                return False

            success = plate_solver.plate_solve()
            self.siril.log("Plate solving completed successfully!" if success
                           else "Plate solving failed", LogColor.GREEN if success else LogColor.SALMON)

            if not success:
                self.siril.log("Check telescope selection (Vespera II/Pro) and DSO name and retry.", LogColor.SALMON)

            return success

        except Exception as e:
            self.siril.log(f"Plate solving error: {e}", LogColor.RED)
            return False

    def _get_simbad_coordinates(self, plate_solver):
        """Get coordinates from SIMBAD with manual fallback."""
        simbad_coords = None
        if plate_solver.dso_name and plate_solver._validate_dso_name(plate_solver.dso_name):
            # Try SIMBAD first
            simbad_coords = plate_solver._query_simbad_coordinates(plate_solver.dso_name)
            if simbad_coords:
                plate_solver.applied_coordinates = simbad_coords
                # Query succeeded – reset failure flag
                self.simbad_query_failed = False
            else:
                # SIMBAD failed – ask user for a manual DSO name
                self.simbad_query_failed = True
                self._request_manual_dso_entry(plate_solver)
        else:
            # No valid DSO name – request manual entry
            self.simbad_query_failed = True
            self._request_manual_dso_entry(plate_solver)

    def _request_manual_dso_entry(self, plate_solver):
        """Request manual DSO entry when automatic extraction fails."""
        self.update_dso_input_visibility.emit()
        self.manual_dso_request.emit()

        manual_dso = self._wait_for_manual_dso_entry()
        if manual_dso and plate_solver._validate_dso_name(manual_dso):
            plate_solver.dso_name = manual_dso
            simbad_coords = plate_solver._query_simbad_coordinates(plate_solver.dso_name)
            if simbad_coords:
                plate_solver.applied_coordinates = simbad_coords

    def _wait_for_manual_dso_entry(self):
        """Wait for manual DSO entry from the main thread."""
        try:
            # Wait for the main thread to provide the DSO name
            self.manual_dso_event.wait(timeout=30.0)  # 30 second timeout

            return self.provided_dso_name

        except Exception as e:
            self.siril.log(f"Waiting for manual DSO entry failed: {e}", LogColor.SALMON)
            return None

    def _get_current_filename(self):
        """Get current image filename."""
        try:
            return self.siril.get_image_filename()
        except Exception:
            return None

    def _run_spcc(self):
        """Run Spectrophotometric Color Correction (SPCC) on the loaded plate‑solved image."""
        filter_name = self.options.get('spcc_filter', 'No Filter').strip()
        self.siril.log(f"SPCC filter selected: {filter_name}", LogColor.BLUE)  # debug

        spcc_sensor = self.options['spcc_sensor']

        if filter_name == "City Light Pollution":
            # CLS filter – use oscfilter
            cmd = f'spcc \"-oscsensor={spcc_sensor}\" \"-oscfilter=Vaonis CLS\"'
            self.siril.log(f"Running SPCC (CLS) with command: {cmd}", LogColor.BLUE)
            self.siril.cmd(cmd)

        elif filter_name == "Dual Band Ha/Oiii":
            # Dual‑band narrowband mode (Ha 656.3 nm, OIII 500.7 nm, 12 nm bandwidth)
            cmd = (
                f'spcc \"-oscsensor={spcc_sensor}\" '
                f'\"-narrowband\" '
                f'\"-rwl=656.3\" \"-rbw=12\" '
                f'\"-gwl=500.7\" \"-gbw=12\"'
            )
            self.siril.log(f"Running SPCC (Dualband) with command: {cmd}", LogColor.BLUE)
            self.siril.cmd(cmd)

        else:  # Default / No Filter
            cmd = (
                f'spcc \"-oscsensor={spcc_sensor}\" '
                f'\"-rfilter=NoFilter\" \"-gfilter=NoFilter\" \"-bfilter=NoFilter\"'
            )
            self.siril.log(f"Running SPCC (No Filter) with command: {cmd}", LogColor.BLUE)
            self.siril.cmd(cmd)

        return True

    def _run_denoise(self):
        """Run denoising based on selected method."""
        method = self.options['denoise_method']

        denoise_commands = {
            'silentium': ("pyscript", "VeraLux_Silentium.py"),
            'graxpert': ("pyscript", "GraXpert-AI.py", "-denoise",
                         f"-strength={self.options.get('denoise_strength', 0.5)}"),
            'cosmic': ("pyscript", "CosmicClarity_Denoise.py")
        }

        if method in denoise_commands:
            self.siril.cmd(*denoise_commands[method])

    def _run_autostretch(self):
        """Run the Siril auto‑stretch command – always linked."""
        # Default values used if not present
        shadowsclip = self.options.get('shadowsclip', -2.8)
        targetbg = self.options.get('targetbg', 0.25)

        cmd_parts = ["autostretch", "-linked",
                     str(shadowsclip), str(targetbg)]

        self.siril.cmd(*cmd_parts)


class VesperaPostprocessingWindow(QMainWindow):
    """Main window for Vespera Postprocessing plugin."""
    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.worker = None
        self.settings = QSettings("VesperaSiril", "Postprocessing")
        self.simbad_query_failed = False

        self.setWindowTitle(f"Vespera Postprocessing v{VERSION}")
        self.setMinimumWidth(400)
        self.setMinimumHeight(700)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        """Build the user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        # Header
        header = QLabel("Vespera Postprocessing")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #88aaff;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Separator
        sep = QFrame()
        sep.setObjectName("Separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        # Background Extraction Group
        bge_group = QGroupBox("Background Extraction")
        bge_layout = QVBoxLayout(bge_group)
        bge_layout.setSpacing(10) 
        self.bge_button_group = QButtonGroup(self)

        self.bge_graxpert = QRadioButton("GraXpert AI (Recommended)")
        self.bge_graxpert.setToolTip(
            "AI-based background extraction.\n"
            "Best for complex gradients and light pollution."
        )
        self.bge_graxpert.setChecked(True)
        self.bge_button_group.addButton(self.bge_graxpert, 0)
        bge_layout.addWidget(self.bge_graxpert)

        # Smoothing slider for GraXpert
        smooth_layout = QHBoxLayout()
        smooth_layout.setContentsMargins(20, 0, 0, 0)
        smooth_label = QLabel("Smoothing:")
        smooth_label.setStyleSheet("color: #888888;")
        smooth_layout.addWidget(smooth_label)

        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(50)
        self.smoothing_slider.setFixedWidth(120)
        smooth_layout.addWidget(self.smoothing_slider)

        self.smoothing_value = QLabel("0.50")
        self.smoothing_value.setFixedWidth(35)
        smooth_layout.addWidget(self.smoothing_value)
        smooth_layout.addStretch()
        bge_layout.addLayout(smooth_layout)

        self.smoothing_slider.valueChanged.connect(
            lambda v: self.smoothing_value.setText(f"{v/100:.2f}")
        )

        self.bge_rbf = QRadioButton("Siril RBF (Fast fallback)")
        self.bge_rbf.setToolTip(
            "Radial Basis Function interpolation.\n"
            "Faster, good for simpler gradients."
        )
        self.bge_button_group.addButton(self.bge_rbf, 1)
        bge_layout.addWidget(self.bge_rbf)

        self.bge_none = QRadioButton("Skip (already extracted)")
        self.bge_button_group.addButton(self.bge_none, 2)
        bge_layout.addWidget(self.bge_none)

        layout.addWidget(bge_group)

        # Calibration Group
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_group)
        cal_layout.setSpacing(10)                          

        # Sensor selection group – now centered
        self.sensor_group = QButtonGroup(self)
        sensor_hbox = QHBoxLayout()
        self.sensor_vesperaII = QRadioButton("Vespera II")
        self.sensor_vesperaPro = QRadioButton("Vespera Pro")
        self.sensor_group.addButton(self.sensor_vesperaII, 0)
        self.sensor_group.addButton(self.sensor_vesperaPro, 1)
        # Default to Pro
        self.sensor_vesperaPro.setChecked(True)

        sensor_hbox.addStretch()
        sensor_hbox.addWidget(self.sensor_vesperaII)
        sensor_hbox.addWidget(self.sensor_vesperaPro)
        sensor_hbox.addStretch()
        cal_layout.addLayout(sensor_hbox)

        # Plate Solve checkbox
        self.plate_solve_cb = QCheckBox("Plate Solve")
        self.plate_solve_cb.setChecked(True)
        self.plate_solve_cb.setToolTip(
            "Attempt plate solving based on file name.\n"
            "DSO name manual entry fallback options.\n"
            "Processing continues even if plate solving fails."
        )
        cal_layout.addWidget(self.plate_solve_cb)

        # Add DSO name input field
        self.dso_input = QLineEdit()
        self.dso_input.setPlaceholderText("Enter DSO name")
        self.dso_input.setToolTip(
            "Manually enter DSO name for plate solving if automatic extraction fails.\n"
            "Examples: M42, IC 342, NGC 7000"
        )
        self.dso_input.setVisible(False)  # Only show when needed
        cal_layout.addWidget(self.dso_input)

        self.plate_solve_cb.stateChanged.connect(self._update_dso_input_visibility)

        # SPCC checkbox
        self.spcc_cb = QCheckBox("Spectrophotometric Color Correction (SPCC)")
        self.spcc_cb.setChecked(True)

        # Force plate solving when SPCC is enabled (SPCC requires plate solving)
        self.spcc_cb.stateChanged.connect(self._on_spcc_state_changed)
        self.spcc_cb.toggled.connect(self._update_spcc_filter_visibility)
        self.spcc_cb.setToolTip(
            "Calibrate colors using Gaia star catalog.\n"
            "Produces accurate, natural star colors."
        )
        cal_layout.addWidget(self.spcc_cb)

        # Uncheck SPCC if Plate Solve is unchecked
        self.plate_solve_cb.stateChanged.connect(self._on_plate_solve_state_changed)

        self.spcc_filter_combo = QComboBox()
        self.spcc_filter_combo.addItems(
            ["No Filter", "Dual Band Ha/Oiii", "City Light Pollution"]
        )
        self.spcc_filter_combo.setCurrentIndex(0)
        cal_layout.addWidget(self.spcc_filter_combo)

        layout.addWidget(cal_group)

        # Denoise Group
        denoise_group = QGroupBox("Denoise")
        denoise_layout = QVBoxLayout(denoise_group)
        denoise_layout.setSpacing(10)

        self.denoise_button_group = QButtonGroup(self)

        self.denoise_none = QRadioButton("None")
        self.denoise_none.setChecked(True)
        self.denoise_button_group.addButton(self.denoise_none, 0)
        denoise_layout.addWidget(self.denoise_none)

        self.denoise_silentium = QRadioButton("VeraLux Silentium (wavelet, PSF‑aware)")
        self.denoise_silentium.setToolTip(
            "Physics‑based wavelet denoiser.\n"
            "Uses actual star geometry for protection."
        )
        self.denoise_button_group.addButton(self.denoise_silentium, 1)
        denoise_layout.addWidget(self.denoise_silentium)

        self.denoise_graxpert = QRadioButton("GraXpert AI")
        self.denoise_graxpert.setToolTip(
            "AI neural network denoiser.\n"
            "Good general‑purpose option.\n"
            "May occasionally add artifacts."
        )
        self.denoise_button_group.addButton(self.denoise_graxpert, 2)
        denoise_layout.addWidget(self.denoise_graxpert)

        self.denoise_cosmic = QRadioButton("Cosmic Clarity")
        self.denoise_cosmic.setToolTip(
            "Alternative AI denoiser with different training.\n"
            "Try if GraXpert produces artifacts."
        )
        self.denoise_button_group.addButton(self.denoise_cosmic, 3)
        denoise_layout.addWidget(self.denoise_cosmic)

        layout.addWidget(denoise_group)

        # Stretch Group 
        stretch_group = QGroupBox("Stretch")
        stretch_layout = QVBoxLayout(stretch_group)
        stretch_layout.setSpacing(10)

        self.stretch_button_group = QButtonGroup(self)

        # "Stretch" radio button
        self.stretch_radio = QRadioButton("AutoStretch")
        self.stretch_radio.setToolTip(
            "Default AutoStretch from SIRIL after preparation is complete."
        )
        self.stretch_radio.setChecked(True)  # default
        self.stretch_button_group.addButton(self.stretch_radio, 0)
        stretch_layout.addWidget(self.stretch_radio)

        # "Veralux HMS" radio button
        self.hms_radio = QRadioButton("VeraLux HMS")
        self.hms_radio.setToolTip(
            "Automatically open HyperMetric Stretch after preparation is complete."
        )
        self.hms_radio.setChecked(False)
        self.stretch_button_group.addButton(self.hms_radio, 1)
        stretch_layout.addWidget(self.hms_radio)

        layout.addWidget(stretch_group)
        # -----------------------------------------

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_label)

        # Process button
        btn_layout = QHBoxLayout()
        self.process_button = QPushButton("Process")
        self.process_button.setObjectName("PrepButton")
        self.process_button.clicked.connect(self._on_process_clicked)
        self.process_button.setStyleSheet("""
            background-color: #285299;
        """)
        btn_layout.addWidget(self.process_button)

        self.close_button = QPushButton("Close") 
        self.close_button.clicked.connect(self.close)
        btn_layout.addWidget(self.close_button)

        layout.addLayout(btn_layout)

    def _load_settings(self):
        """Load saved settings."""
        bge = self.settings.value("bge_method", 0, type=int)
        self.bge_button_group.button(bge).setChecked(True)

        smoothing = self.settings.value("smoothing", 50, type=int)
        self.smoothing_slider.setValue(smoothing)

        self.plate_solve_cb.setChecked(
            self.settings.value("plate_solve", True, type=bool))
        self.spcc_cb.setChecked(
            self.settings.value("spcc", True, type=bool))

        # Load sensor selection
        sensor_id = self.settings.value("sensor", 1, type=int)
        if sensor_id == 0:
            self.sensor_vesperaII.setChecked(True)
        else:
            self.sensor_vesperaPro.setChecked(True)

        denoise = self.settings.value("denoise_method", 0, type=int)
        self.denoise_button_group.button(denoise).setChecked(True)

        stretch_mode = self.settings.value("stretch_mode", 0, type=int)
        if stretch_mode == 0:
            self.stretch_radio.setChecked(True)
        else:
            self.hms_radio.setChecked(True)

        self.spcc_filter_combo.setCurrentIndex(
            self.settings.value("spcc_filter_index", 0, type=int))

    def _save_settings(self):
        """Save current settings."""
        self.settings.setValue("bge_method", self.bge_button_group.checkedId())
        self.settings.setValue("smoothing", self.smoothing_slider.value())
        self.settings.setValue("plate_solve", self.plate_solve_cb.isChecked())
        self.settings.setValue("spcc", self.spcc_cb.isChecked())
        # Save sensor selection
        self.settings.setValue("sensor", self.sensor_group.checkedId())
        self.settings.setValue("denoise_method", self.denoise_button_group.checkedId())
        self.settings.setValue("stretch_mode", 0 if self.stretch_radio.isChecked() else 1)
        self.settings.setValue("spcc_filter_index", self.spcc_filter_combo.currentIndex()) 

    def _get_options(self):
        """Collect current options into a dictionary."""
        bge_id = self.bge_button_group.checkedId()
        bge_methods = {0: 'graxpert', 1: 'siril_rbf', 2: 'none'}

        denoise_id = self.denoise_button_group.checkedId()
        denoise_methods = {0: 'none', 1: 'silentium', 2: 'graxpert', 3: 'cosmic'}

        sensor_id = self.sensor_group.checkedId()
        # Map sensor to pixel size and SPCC sensor
        if sensor_id == 0:   # Vespera II
            pixel_size_um = 2.9
            spcc_sensor = "Sony IMX585"
        else:                # Vespera Pro
            pixel_size_um = 2.00
            spcc_sensor = "Sony IMX676"

        return {
            'bge_method': bge_methods.get(bge_id, 'graxpert'),
            'bge_smoothing': self.smoothing_slider.value() / 100.0,
            'plate_solve': self.plate_solve_cb.isChecked(),
            'spcc': self.spcc_cb.isChecked(),
            'spcc_filter': self.spcc_filter_combo.currentText(),
            'denoise_method': denoise_methods.get(denoise_id, 'none'),
            'denoise_strength': 0.5,
            'autostretch': self.stretch_radio.isChecked(),
            'hms_stretch': self.hms_radio.isChecked(),
            'optimize_format': True,
            'continue_on_failure': True,
            'pixel_size_um': pixel_size_um,
            'spcc_sensor': spcc_sensor,
        }

    def _on_plate_solve_state_changed(self, state):
        """If Plate Solve is unchecked while SPCC is checked,
           automatically uncheck SPCC (SPCC requires plate solving)."""
        if state == Qt.CheckState.Unchecked.value and self.spcc_cb.isChecked():
            self.spcc_cb.blockSignals(True)
            self.spcc_cb.setChecked(False)
            self.spcc_cb.blockSignals(False)

        self._update_spcc_filter_visibility()

    def _on_spcc_state_changed(self, state: int):
        """Enable plate solving when SPCC is enabled; do nothing when SPCC is disabled."""
        if state == Qt.CheckState.Checked.value:
            if not self.plate_solve_cb.isChecked():
                self.plate_solve_cb.setChecked(True)
                self.siril.log("Plate solving enabled (required for SPCC)", LogColor.BLUE)

        # No need to enforce linked mode – auto‑stretch is always linked
        self._update_spcc_filter_visibility()

    def _update_spcc_filter_visibility(self, checked: bool | None = None):
        """
        Show the filter combo only when SPCC is enabled *and* Plate Solve
        is also checked.  The argument `checked` comes from the SPCC
        checkbox toggle; if None we use its current state.
        """
        if checked is None:
            checked = self.spcc_cb.isChecked()
        # Visible only when both SPCC and Plate Solve are active
        visible = checked and self.plate_solve_cb.isChecked()
        self.spcc_filter_combo.setVisible(visible)

    def _update_dso_input_visibility(self):
        """Show DSO input when needed for plate solving."""
        show_input = (self.plate_solve_cb.isChecked() and
                      (not self._has_valid_filename() or 
                       getattr(self, 'simbad_query_failed', False)))

        self.dso_input.setVisible(show_input)

        if show_input:
            self.siril.log("Please enter DSO name for plate solving", LogColor.BLUE)

    def _has_valid_dso_name(self):
        """Check if we have a valid extracted DSO name."""
        # Check if plate solver exists and has a valid DSO name
        if hasattr(self, 'plate_solver') and self.plate_solver and self.plate_solver.dso_name:
            return len(self.plate_solver.dso_name.strip()) > 0
        return False

    def _has_valid_filename(self):
        """Check if we have a valid filename for extraction."""
        try:
            filename = self._get_current_filename()
            return filename is not None and len(filename.strip()) > 0
        except:
            return False

    def _get_current_filename(self):
        """Get current image filename."""
        try:
            return self.siril.get_image_filename()
        except Exception:
            return None

    def _on_process_clicked(self):
        """Handle Prep button click."""
        # Check if an image is loaded
        try:
            img_shape = self.siril.get_image_shape()
            if img_shape is None:
                QMessageBox.warning(self, "No Image",
                    "Please load a Vespera TIFF image first.")
                return
        except Exception as e:
            QMessageBox.warning(self, "No Image",
                "Please load a Vespera TIFF image first.")
            return

        self._save_settings()
        options = self._get_options()

        # Validate at least one operation selected
        if (options['bge_method'] == 'none' and
            not options['plate_solve'] and
            not options['spcc'] and
            options['denoise_method'] == 'none' and
            not options['autostretch'] and
            not options['hms_stretch']):
            QMessageBox.information(self, "Nothing to do",
                "Please select at least one operation.")
            return

        # Disable UI during processing
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start worker thread
        dso_name = self.dso_input.text().strip() if self.dso_input.isVisible() else None
        self.worker = PrepWorker(self.siril, options, dso_name)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.manual_dso_request.connect(self._on_manual_dso_request)
        self.worker.update_dso_input_visibility.connect(self._update_dso_input_visibility)
        self.worker.start()

    def _on_progress(self, percent, message):
        """Handle progress updates."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_manual_dso_request(self):
        """Handle manual DSO request from worker thread."""
        try:
            # Show input dialog in main thread
            dso_name, ok = QInputDialog.getText(
                self,
                "Manual DSO Entry Required",
                "SIMBAD query failed. Please enter DSO name:",
                QLineEdit.EchoMode.Normal,
                ""
            )

            if ok and dso_name.strip():
                self.siril.log(f"Using manual DSO entry: {dso_name}", LogColor.BLUE)
                # Store the manual DSO name and signal the worker
                if self.worker:
                    self.worker.provided_dso_name = dso_name.strip()
            else:
                self.siril.log("Manual DSO entry cancelled", LogColor.SALMON)
                # Signal the worker that no DSO name was provided
                if self.worker:
                    self.worker.provided_dso_name = None

            # Always signal the worker to continue
            if self.worker:
                self.worker.manual_dso_event.set()

        except Exception as e:
            self.siril.log(f"Manual DSO entry failed: {e}", LogColor.SALMON)
            # Signal the worker that an error occurred
            if self.worker:
                self.worker.provided_dso_name = None
                self.worker.manual_dso_event.set()

    def _on_finished(self, success, message):
        """Handle completion."""
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success:
            
            try:
                orig_name = self.siril.get_image_filename()
                if orig_name:
                    dir_path, base = os.path.split(orig_name)
                    name, ext = os.path.splitext(base)

                    new_name = f'"{name}_processed"'

                    self.siril.cmd("save", new_name)
            except Exception as e:
                self.siril.log(f"Could not save processed image: {e}",
                               LogColor.SALMON)

            
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: #88ff88;")

            # Launch HMS if requested
            if self.hms_radio.isChecked():
                try:
                    self.siril.cmd("pyscript", "VeraLux_HyperMetric_Stretch.py")
                    self.close()
                except Exception as e:
                    self.siril.log(f"Could not launch HMS: {e}", LogColor.SALMON)
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #ff8888;")
            QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        """Handle window close."""
        self._save_settings()
        event.accept()


def main():
    """Main entry point."""
    siril = SirilInterface()

    try:
        siril.connect()
        siril.log("Vespera Postprocessing started", LogColor.GREEN)

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = VesperaPostprocessingWindow(siril)
        window.show()

        app.exec()
    except Exception as e:
        siril.log(f"Error: {e}", LogColor.RED)
        raise
    finally:
        siril.disconnect()


if __name__ == "__main__":
    main()
