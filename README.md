# FTP FITS / TIFF Downloader for Vespera Smart Telescope

---

## 📦 What It Does  
The script `sync_vespera.py` connects to the Vespera Smart Telescope’s FTP server, finds the most recently modified directory under a given root, and downloads all `.fits` and/or `.tif` files to your local machine.  
Key features:

- **Progress tracking** – per‑file and overall progress bars with speed estimates.  
- **Automatic renaming & sorting** – files are moved into object‑specific folders with a consistent naming scheme.  
- **Optional server cleanup** – delete files from the FTP server after download (`--delete` flag).  
- **Two modes** – a quick “original” mode (`python sync_vespera.py --cli`) and an interactive enhanced CLI that prompts for file types, destination folder, and deletion preference.

---

## ⚙️ Prerequisites  

| Requirement | Minimum Version |
|-------------|-----------------|
| Python | 3.8+ |
| Standard library only – no external dependencies |

---

## 🚀 Installation & Usage  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/gtrainar/vespera.git
   cd vespera
   ```

2. **Run the script**

| Mode | Command |
|------|---------|
| **Quick original mode** (no prompts, uses defaults) | `python sync_vespera.py --cli` |
| **Enhanced interactive mode** (default) | `python sync_vespera.py` |

> In the enhanced mode you’ll be asked to choose between TIFF, FITS or both, specify a destination directory (default is `~/Vespera`), and decide whether to delete files from the server after download.

3. **Optional: Delete files on the server**  
   In interactive mode type `y` when prompted for “Delete files from server after download?”  
   In quick mode you can add the `--delete` flag:  
   ```bash
   python sync_vespera.py --cli --delete
   ```

---

## 📄 Configuration  

The script uses a `Config` class that can be overridden from the command line or by editing the source.  
Default values are shown below:

```python
DEFAULT_CONFIG = {
    "FTP_SERVER": "10.0.0.1",
    "FTP_USER": "anonymous", 
    "FTP_PASSWORD": "",
    "REMOTE_DIR": "/user",
    "LOCAL_DIR": Path("/Users/Astro/Photo/Vespera"),
    "CHECK_INTERVAL": 1800,
    "MAX_FAILED_CHECKS": 10,
    "FILE_TYPES": ('.fits', '.tif'),
    "DELETE_AFTER_DOWNLOAD": False
}
```

- **FTP_SERVER** – Address of the Vespera FTP host.  
- **FILE_TYPES** – Tuple of extensions to download (`.fits`, `.tif`).  
- **LOCAL_DIR** – Where downloaded files will be stored.  
- **DELETE_AFTER_DOWNLOAD** – Set to `True` if you want files removed from the server after a successful transfer.

---

## 🏁 Example Output  

```text
Vespera FTP Downloader - Enhanced CLI Mode
==========================================
File Types to Download:
1. TIFF files (.tif)
2. FITS files (.fits)
3. Both TIFF and FITS
Select file types (1-3) [1]: 3
Destination directory [/Users/Astro/Photo/Vespera]: 
Delete files from server after download? (y/N): n

Configuration:
  Server: 10.0.0.1
  User: anonymous
  File types: .tif, .fits
  Destination: /Users/Astro/Photo/Vespera
  Delete after download: False

Exploring: /user/2026-01-20_Observation
Starting download of 42 files...
Downloading img1.fits [=====                         ] 12.3%  45.6 KB/s
...
Completed 42/42 files (100.0%)
✅ Download completed successfully!
📊 Summary:
   Files downloaded: 42
   Total files found: 42
   Time taken: 45.3 seconds
```

Press **Ctrl + C** at any time to cancel the operation.

---

## 📄 License  

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---
