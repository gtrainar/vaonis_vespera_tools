# Vespera‑Suite  
**A collection of scripts for the Vaonis Vespera Smart Telescope**

> **Workflow order:**  
> 1. `sync_vespera.py` – FTP downloader (run first).  
> 2. `Vespera_Preprocessing.py` – Raw FITS preprocessing, stacking, and post-stacking pipeline.

---

## Table of Contents
1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - 4.1 [FTP Downloader](#ftp-downloader)  
   - 4.2 [Preprocessing](#preprocessing)  
5. [Configuration & Settings](#configuration-settings)  
6. [Examples](#examples)  
7. [Troubleshooting](#troubleshooting)  
8. [License](#license)  
9. [Contributing](#contributing)  

---

## Features

| Script | What it does | Key Options |
|--------|--------------|-------------|
| **sync_vespera.py** | FTP client that finds the most recent observation folder, downloads FITS/TIFF files, renames them, moves to an object‑based directory tree and optionally deletes the originals. | • CLI or interactive mode <br>• Progress bar <br>• Cancel with Ctrl‑C |
| **Vespera_Preprocessing.py** | Detects dark/light frames, calibrates, registers (drizzle or standard), stacks, and runs a full post-stacking pipeline (plate-solving, SPCC colour calibration, auto-stretch) outputting a 32‑bit FITS ready for viewing. | • `Bayer Drizzle` (recommended) <br>• Feathering 0–50 px <br>• Two‑pass registration <br>• Batch processing <br>• SPCC with filter presets <br>• Auto-stretch <br>• Clean temporary files |

---

## Prerequisites

| Component | Minimum Version |
|-----------|-----------------|
| **Python** | 3.10+ |
| **Siril** | 1.4+ (with `sirilpy` plugin) |
| **PyQt6** | 6.x |
| **astropy** | any recent release |

> All dependencies are automatically installed when the script first runs via `sirilpy.ensure_installed("PyQt6", "astropy")`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/gtrainar/vespera-suite.git
cd vespera-suite

# (Optional) Create a virtual environment if you run scripts from a console
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install Python dependencies
pip install PyQt6 astropy
```

> The preprocessing script is designed to be dropped into the Siril `scripts` folder.  
> The FTP downloader can be run directly from a terminal.

---

## Usage

### 4.1 FTP Downloader  
Run from the command line:

```bash
python sync_vespera.py          # interactive mode (default)
python sync_vespera.py --cli    # original CLI mode
```

You'll be prompted for:

* File type(s) to download (TIFF, FITS, or both).  
* Destination directory.  
* Whether to delete the files on the server after download.

The script prints a progress bar per file and shows a summary at the end.  
Press **Ctrl‑C** to cancel; partial files are removed automatically.

---

### 4.2 Preprocessing

Open *Siril* → **Scripts** → `Vespera_Preprocessing.py`.

The plugin runs a **complete pipeline** in a single click: dark calibration → registration → stacking → plate-solving → colour calibration → auto-stretch. Every stage is optional and configurable from the GUI.

---

#### Folder Structure – No Re‑organisation Needed!

The plugin **automatically detects** how your Vespera exported the data. Just point it at your observation folder.

**Supported layouts**

```
# Native Vespera export (flat structure)
Vespera_Observation_Folder/
├── img-0001-dark.fits          ← single dark frame
├── 01-images-initial/
│   ├── img-0001.fits           ← light frames
│   ├── img-0002.fits
│   └── …
└── [TIFF previews, JSON metadata, …]

# Organised structure
observation_folder/
├── darks/
│   └── dark_000001.fit
└── lights/
    ├── light_000001.fit
    └── light_000002.fit

# Flat structure (auto-organised on first run)
observation_folder/
├── master_dark.fits            ← detected by filename pattern
├── frame_000001.fits
├── frame_000002.fits
└── …
```

TIFF preview files found alongside the light frames are moved automatically to a `reference/` sub-folder so they don't interfere with processing.

---

#### Running the Plugin

1. Open Siril and navigate to your observation folder.  
2. Go to **Scripts** → **Vespera_Preprocessing**.  
3. Configure the options in the GUI (see sections below).  
4. Click **Start Processing** – a progress bar and colour-coded log track every stage.

---

#### Stacking Options

| Option | Description |
|--------|-------------|
| **Sky Quality** | Bortle 1–2 through 7–8. Sets sigma-clipping thresholds (σ_low / σ_high) automatically. |
| **Stacking Method** | See table below. |
| **Background Extraction** | Fits and subtracts a smooth RBF sky background from each calibrated frame before registration. Useful on light-polluted or gradient-heavy skies. |
| **2-Pass Registration** | Runs a second alignment pass with `-framing=max`, preserving the maximum common field of view. Best with large sessions (50+ frames). |
| **Feathering** | Blends stacked sub-image edges (0–50 px) to soften hard seam artefacts in mosaics or high-rotation sessions. |
| **Batch Processing** | Splits lights into chunks of N frames, stacks each independently, then combines. Reduces peak disk usage for large sessions. Minimum recommended chunk size: **20 frames**. |

**Stacking methods**

| Method | Best For | Technical Details |
|--------|----------|-------------------|
| **Bayer Drizzle (Recommended)** | Most sessions | Gaussian kernel + area interpolation. Handles 10–15° field rotation without moiré. |
| **Bayer Drizzle (Square)** | Photometry | Classic HST square kernel; mathematically flux-preserving. |
| **Bayer Drizzle (Nearest)** | Checkerboard artefacts | Nearest-neighbour interpolation; zero CFA boundary artefacts. |
| **Standard Registration** | Short sessions < 30 min | Classic debayer → register, no drizzle. Faster, lower RAM. |
| **Drizzle 2× Upscale** | Maximum resolution | Doubles output to 7072 × 7072 px. Requires 50+ well-dithered frames. |

---

#### Post-Stacking Options

These steps run automatically on the final stacked image.

| Option | Description |
|--------|-------------|
| **SPCC** | Spectrophotometric Color Calibration using the Gaia DR3 catalogue. Produces photometrically accurate, natural star colours. Requires an active internet connection (SIMBAD + Gaia). **Plate-solving runs automatically beforehand**; if it fails, SPCC is skipped. |
| **SPCC Filter** | `No Filter` – broadband, uses sensor spectral response only. `City Light Pollution (CLS)` – applies the Vaonis CLS transmission curve. `Dual Band Ha/Oiii` – narrowband mode, Hα 656.3 nm / [O III] 500.7 nm (12 nm BW each). |
| **Auto-Stretch** | Applies Siril's linked Midtone Transfer Function (shadows clip −2.8σ, target background 0.25). The saved FITS is immediately viewable. ⚠ The stretch is non-reversible — disable if you want to keep the linear stack for further processing. |
| **Clean Temporary Files** | Deletes `process/`, `masters/`, and `final_stack/` after a successful run. Reclaims several GB of intermediate data. ⚠ Leave off until you are satisfied with the result. |

---

#### Output Files

| File | Description |
|------|-------------|
| `result_XXXXs.fit` | Final stacked image (32-bit, linear or stretched depending on settings). The filename embeds the total integration time in seconds. |
| `final_stacked_batch.fit` | Final image when Batch Processing is enabled. |
| `masters/dark_stacked.fit` | Master dark frame used for calibration. |
| `logs/disk_usage_TIMESTAMP.log` | Disk free/used/directory-size log sampled every 5 s during processing. |
| `logs/siril_console_TIMESTAMP.log` | Full colour-coded Siril console output for the run, including the configuration summary. |
| `process/` | Intermediate calibrated and registered frames (deleted if Clean Temporary Files is enabled). |
| `reference/` | TIFF preview files moved out of the lights folder before processing. |

---

## Configuration & Settings

| File | What it configures |
|------|--------------------|
| `Vespera_Preprocessing.py` | Sky presets, stacking methods, feathering distance, two-pass flag, batch size, SPCC filter, auto-stretch parameters, clean-temp flag |
| `sync_vespera.py` | FTP server / user / password, remote & local directories, file types, delete-after-download flag |

All GUI settings are persisted automatically via Qt's `QSettings` and restored on the next launch.

---

## Examples

```bash
# Download the newest observation set and delete it from the server
python sync_vespera.py --cli

# Preprocess a folder with 1 dark and 200 lights in batch mode
# (configure batch size and other options in the GUI, then click Start Processing)
python Vespera_Preprocessing.py
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ImportError: No module named 'sirilpy'` | Script not running inside Siril | Open the script from inside Siril via the Scripts menu. |
| "No dark frames found" | Dark file missing or not named correctly | Ensure a `*-dark.fits` file exists alongside the lights, or create a `darks/` folder. |
| Plate solve fails | DSO not found in FITS header, or SIMBAD timed out | The `OBJECT` keyword is read from the first light frame. Check it with a FITS viewer; SPCC will be skipped gracefully if solving fails. |
| SPCC skipped with no error | Plate solve failed silently | Check `logs/siril_console_*.log` for the plate-solve output. Verify internet connectivity. |
| FTP connection refused | Wrong credentials or server address | Verify `FTP_SERVER`, `FTP_USER`, and `FTP_PASSWORD` in `sync_vespera.py`. |
| GUI freezes | Long-running Siril command | Processing runs in a background `QThread`; a true freeze suggests a Siril crash. Check the console log. |
| Stacking fails on small chunks | Too few frames for sigma-clip rejection | Increase batch chunk size. Minimum recommended: **20 frames per chunk**. |
| 2-Pass registration artefacts | Chunk too small to compute a reliable reference | Use 2-Pass only with chunks of 50+ frames, or disable it for batch runs. |

---

## License

MIT © 2025 G. Trainar  
See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repo.  
2. Create a feature branch (`git checkout -b feat/…`).  
3. Commit and push.  
4. Open a Pull Request.

All contributions are welcome — especially improvements to the GUI, new stacking methods, or additional plate-solving backends.
