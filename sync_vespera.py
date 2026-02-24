#!/usr/bin/env python3
"""
FTP FITS / TIFF downloader from Vespera Smart Telescope with CLI interface
Author:  G. Trainar
Date:    2025‑11‑27
Modified: 2026-01-20 - Refactored to use CLI with progress updates
"""

from __future__ import annotations

import ftplib
import os
import re
import shutil
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Callable, Any

# --------------------------------------------------------------------------- #
#  Configuration with defaults
# --------------------------------------------------------------------------- #

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

class Config:
    def __init__(self, **kwargs):
        self._config = DEFAULT_CONFIG.copy()
        self._config.update(kwargs)

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)

    def update(self, **kwargs):
        self._config.update(kwargs)

# --------------------------------------------------------------------------- #
#  Progress Tracking Class for CLI
# --------------------------------------------------------------------------- #

class CLIProgressTracker:
    def __init__(self):
        self.total_files = 0
        self.downloaded_files = 0
        self.current_file = ""
        self.current_progress = 0
        self.cancelled = False
        self.start_time = None
        # Per‑file timing for accurate speed calculation
        self.current_file_start: Optional[float] = None
        self.last_bytes_downloaded: int = 0
        # Track last printed line length to clear properly
        self.last_line_len: int = 0

    def start_operation(self, total_files: int):
        self.total_files = total_files
        self.downloaded_files = 0
        self.cancelled = False
        self.start_time = time.time()
        print(f"Starting download of {total_files} files...")

    def update_file_progress(self, filename: str, bytes_downloaded: int, total_bytes: int):
        # Detect new file and reset per‑file counters
        if filename != self.current_file:
            self.current_file_start = time.time()
            self.last_bytes_downloaded = 0
        self.current_file = filename
        self.current_progress = bytes_downloaded / total_bytes * 100 if total_bytes > 0 else 0
        
        # Calculate per‑file average speed using cumulative bytes and elapsed time since file start
        elapsed_file = time.time() - (self.current_file_start or time.time())
        speed = bytes_downloaded / elapsed_file if elapsed_file > 0 else 0
        self.last_bytes_downloaded = bytes_downloaded
        speed_str = f"{speed/1024:.1f} KB/s" if speed > 0 else "Calculating..."
        
        # Update progress display
        progress_bar = self._create_progress_bar(self.current_progress)
        line = f"\rDownloading {filename} {progress_bar} {self.current_progress:.1f}% {speed_str}"
        # Clear previous longer line if needed
        if len(line) < self.last_line_len:
            line += ' ' * (self.last_line_len - len(line))
        print(line, end="", flush=True)
        self.last_line_len = len(line)
        if len(line) < self.last_line_len:
            line += ' ' * (self.last_line_len - len(line))
        print(line, end="", flush=True)
        self.last_line_len = len(line)

    def file_completed(self):
        self.downloaded_files += 1
        self.current_progress = 0
        # Reset per‑file timing for next file
        self.current_file_start = None
        self.last_bytes_downloaded = 0
        overall_progress = (self.downloaded_files / self.total_files * 100) if self.total_files > 0 else 0
        print(f"\rCompleted {self.downloaded_files}/{self.total_files} files ({overall_progress:.1f}%)")

    def cancel(self):
        self.cancelled = True
        print("\rDownload cancelled by user.")

    def _create_progress_bar(self, progress: float, width: int = 30) -> str:
        """Create a simple progress bar for CLI"""
        filled = int(width * progress / 100)
        remaining = width - filled
        return f"[{'=' * filled}{' ' * remaining}]"

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #

def list_dir_details(ftp: ftplib.FTP, path: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Return a list of (name, facts) tuples for the given remote directory.
    Uses MLSD if available; otherwise falls back to parsing `dir` output.

    :param ftp:   Connected ftplib.FTP instance
    :param path:  Remote directory to list
    :return:      List of (name, facts) tuples
    """
    try:
        # MLSD gives us a dictionary of facts (type, modify, size, …)
        return list(ftp.mlsd(path))
    except Exception:
        # Fallback: parse the output of `dir`
        lines: List[str] = []

        def collect(line: str) -> None:
            lines.append(line)

        ftp.dir(path, collect)
        entries: List[Tuple[str, Dict[str, str]]] = []

        for line in lines:
            # Typical format: drwxr-xr-x 1 owner group size month day time name
            parts = line.split(maxsplit=8)
            if len(parts) < 9:
                continue
            perms, _, _, _, size, month, day, time_or_year, name = parts
            entry_type = "dir" if perms[0] == "d" else "file"
            entries.append((name, {"type": entry_type}))
        return entries


def find_most_recent_dir(ftp: ftplib.FTP, remote_root: str) -> str:
    """
    Find the most recently modified sub‑directory under `remote_root`.

    :param ftp:          Connected ftplib.FTP instance
    :param remote_root:  Remote directory to search in
    :return:             Name of the most recent sub‑directory
    """
    entries = list_dir_details(ftp, remote_root)
    dirs: List[Tuple[str, Dict[str, str]]] = [
        (name, facts) for name, facts in entries if facts.get("type") == "dir"
    ]

    if not dirs:
        raise RuntimeError(f"No directories found in {remote_root}")

    # Prefer MLSD modify time; if not available, just pick the first one
    def get_modify_time(name: str, facts: Dict[str, str]) -> datetime | None:
        mod = facts.get("modify")
        if mod:
            # MLSD modify format: YYYYMMDDHHMMSS
            return datetime.strptime(mod, "%Y%m%d%H%M%S")
        return None

    dirs_with_time = [
        (name, get_modify_time(name, facts)) for name, facts in dirs
    ]

    # Filter out entries without a modify time
    dirs_with_time = [(n, t) for n, t in dirs_with_time if t is not None]

    if dirs_with_time:
        # Sort by modify time descending
        most_recent = sorted(dirs_with_time, key=lambda x: x[1], reverse=True)[0][0]
    else:
        # Fallback – just take the first directory in the listing
        most_recent = dirs[0][0]

    return most_recent


def download_file_with_progress(
    ftp: ftplib.FTP,
    remote_path: str,
    local_path: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None
) -> bool:
    """
    Download a single file with progress tracking and cancellation support
    """
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Get file size
        try:
            file_size = ftp.size(remote_path)
        except:
            file_size = 0

        bytes_downloaded = 0
        start_time = time.time()

        def callback(data):
            nonlocal bytes_downloaded
            if cancel_event and cancel_event.is_set():
                raise Exception("Download cancelled by user")

            # Write data in append mode to avoid overwriting
            with open(local_path, 'ab') as f:
                f.write(data)
            bytes_downloaded += len(data)

            if progress_callback and file_size and file_size > 0:
                progress_callback(local_path.name, bytes_downloaded, file_size)

        # Ensure file doesn't exist before starting download
        if local_path.exists():
            local_path.unlink()
            
        ftp.retrbinary(f"RETR {remote_path}", callback)

        return True

    except Exception as e:
        if local_path.exists():
            local_path.unlink()  # Clean up partial download
        return False


def count_files_recursively(ftp: ftplib.FTP, remote_path: str, config: Config) -> int:
    """
    Count all files recursively that match the file type criteria
    """
    file_pattern = r"\.(" + "|".join(ext.lstrip('.') for ext in config["FILE_TYPES"]) + r")$"
    file_regex = re.compile(file_pattern, re.IGNORECASE)
    
    total_count = 0
    
    def _count_in_directory(path: str) -> int:
        count = 0
        try:
            entries = list_dir_details(ftp, path)
            
            # Count files in current directory
            files = [
                name for name, facts in entries
                if facts.get("type") == "file" and file_regex.search(name)
            ]
            count += len(files)
            
            # Recursively count in subdirectories
            subdirs = [name for name, facts in entries if facts.get("type") == "dir"]
            for sub in subdirs:
                sub_path = f"{path}/{sub}"
                count += _count_in_directory(sub_path)
                
        except Exception:
            pass  # Skip directories that can't be accessed
            
        return count
    
    return _count_in_directory(remote_path)


def download_and_process_dir(
    ftp: ftplib.FTP,
    remote_path: str,
    config: Config,
    progress_tracker: Optional[CLIProgressTracker] = None,
    cancel_event: Optional[threading.Event] = None
) -> bool:
    """
    Enhanced version with file type filtering, progress tracking, and deletion support
    """
    found_files = False

    # Create file type regex pattern
    file_pattern = r"\.(" + "|".join(ext.lstrip('.') for ext in config["FILE_TYPES"]) + r")$"
    file_regex = re.compile(file_pattern, re.IGNORECASE)

    entries = list_dir_details(ftp, remote_path)

    # Filter files based on selected types
    files_to_download = [
        name for name, facts in entries
        if facts.get("type") == "file" and file_regex.search(name)
    ]

    # Count total files for progress tracking is now done in the main function

    # Process files with progress tracking
    for filename in files_to_download:
        if cancel_event and cancel_event.is_set():
            break

        local_file = config["LOCAL_DIR"] / filename
        remote_file_path = f"{remote_path}/{filename}"

        # Download with progress
        download_success = download_file_with_progress(
            ftp, remote_file_path, local_file,
            lambda filename_param, bytes_downloaded, total_bytes: (
                progress_tracker.update_file_progress(
                    filename_param, bytes_downloaded, total_bytes
                )
            ) if progress_tracker else None,
            cancel_event
        )

        if download_success:
            # Process and rename file (existing logic)
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", remote_path)
            date = date_match.group(0) if date_match else "unknown"

            obj_match = re.search(r"observation[-_]([^/]+)", remote_path)
            objet = obj_match.group(1).upper() if obj_match else "UNKNOWN"
            if any(char.isdigit() for char in objet):
                objet = objet[:objet.find(next(filter(str.isdigit, objet)))] + " " + objet[objet.find(next(filter(str.isdigit, objet))):]

            if filename.lower().endswith(".fits"):
                # Dark file special case – keep it under the object folder
                if "-dark.fits" in filename:
                    dest_dir = config["LOCAL_DIR"] / objet / "darks"
                else:
                    dest_dir = config["LOCAL_DIR"] / objet / "lights"
            else:
                # Keep original behaviour for non‑fits files
                date_match = re.search(r"\d{4}-\d{2}-\d{2}", remote_path)
                date = date_match.group(0) if date_match else "unknown"

                obj_match = re.search(r"observation[-_]([^/]+)", remote_path)
                objet = obj_match.group(1).upper() if obj_match else "UNKNOWN"
                if any(char.isdigit() for char in objet):
                    idx = next(i for i, c in enumerate(objet) if c.isdigit())
                    objet = f"{objet[:idx]} {objet[idx:]}"

                dest_dir = config["LOCAL_DIR"] / objet

            dest_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(filename).suffix.lstrip(".")
            base_name = f"{objet}_{date}"
            new_name = f"{base_name}-{filename.split('-')[1].split('.')[0]}.{ext}"

            if "-dark.fits" in filename:
                new_name = "master_dark.fits"

            dest_path = dest_dir / new_name
            shutil.move(str(local_file), str(dest_path))

            # Delete from server if requested
            if config["DELETE_AFTER_DOWNLOAD"]:
                try:
                    ftp.delete(remote_file_path)
                except Exception as e:
                    print(f"Warning: Could not delete {remote_file_path}: {e}")

            if progress_tracker:
                progress_tracker.file_completed()

            found_files = True

    # Recursive processing of subdirectories
    subdirs = [name for name, facts in entries if facts.get("type") == "dir"]
    for sub in subdirs:
        if cancel_event and cancel_event.is_set():
            break
        sub_path = f"{remote_path}/{sub}"
        if download_and_process_dir(
            ftp, sub_path, config, progress_tracker, cancel_event
        ):
            found_files = True

    return found_files


# --------------------------------------------------------------------------- #
#  Main routine
# --------------------------------------------------------------------------- #

def original_main_function():
    """Original main function logic (renamed for hybrid support)"""
    global FAILED_CHECKS

    config = Config()  # Use default configuration

    # Ensure local base directory exists
    config["LOCAL_DIR"].mkdir(parents=True, exist_ok=True)

    try:
        ftp = ftplib.FTP(config["FTP_SERVER"])
        ftp.login(user=config["FTP_USER"], passwd=config["FTP_PASSWORD"])
    except Exception as exc:
        print(f"❌ Could not connect to {config['FTP_SERVER']}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        most_recent = find_most_recent_dir(ftp, config["REMOTE_DIR"])
        print(f"Exploring most recent directory: {config['REMOTE_DIR']}/{most_recent}")

        found = download_and_process_dir(ftp, f"{config['REMOTE_DIR']}/{most_recent}", config)

        if not found:
            FAILED_CHECKS += 1
            print(f"⚠️ No .fits/.tif files found. Failed attempt #{FAILED_CHECKS}")
        else:
            FAILED_CHECKS = 0
            print("✅ Files downloaded successfully.")

        if FAILED_CHECKS >= config["MAX_FAILED_CHECKS"]:
            print(
                f"❌ Maximum of {config['MAX_FAILED_CHECKS']} consecutive failures reached. "
                "Stopping script."
            )
    finally:
        ftp.quit()


def enhanced_cli_mode():
    """Enhanced CLI mode with user prompts and progress tracking"""
    print("Vespera FTP Downloader - Enhanced CLI Mode")
    print("=" * 50)
    
    # Get user preferences
    print("File Types to Download:")
    print("1. TIFF files (.tif)")
    print("2. FITS files (.fits)")
    print("3. Both TIFF and FITS")

    while True:
        file_choice = input("Select file types (1-3) [1]: ").strip()
        if not file_choice:  # Default to 1 if no input
            file_choice = "1"
            break
        if file_choice in ["1", "2", "3"]:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    file_types = []
    if file_choice in ["1", "3"]:
        file_types.append('.tif')
    if file_choice in ["2", "3"]:
        file_types.append('.fits')
    
    # Get destination directory
    dest_dir = input(f"Destination directory [{DEFAULT_CONFIG['LOCAL_DIR']}]: ").strip()
    if not dest_dir:
        dest_dir = str(DEFAULT_CONFIG["LOCAL_DIR"])
    
    # Validate destination
    dest_path = Path(dest_dir)
    if not dest_path.exists():
        try:
            dest_path.mkdir(parents=True)
            print(f"Created directory: {dest_path}")
        except Exception as e:
            print(f"❌ Could not create directory {dest_path}: {e}")
            sys.exit(1)
    elif not dest_path.is_dir():
        print(f"❌ {dest_path} is not a directory")
        sys.exit(1)
    
    # Get advanced options
    delete_after = input("Delete files from server after download? (y/N): ").strip().lower() == 'y'
    
    # Create config
    config = Config(
        FILE_TYPES=tuple(file_types),
        LOCAL_DIR=dest_path,
        DELETE_AFTER_DOWNLOAD=delete_after
    )
    
    print(f"\nConfiguration:")
    print(f"  Server: {config['FTP_SERVER']}")
    print(f"  User: {config['FTP_USER']}")
    print(f"  File types: {', '.join(config['FILE_TYPES'])}")
    print(f"  Destination: {config['LOCAL_DIR']}")
    print(f"  Delete after download: {config['DELETE_AFTER_DOWNLOAD']}")
    print()
    
    # Start download with progress tracking
    progress_tracker = CLIProgressTracker()
    cancel_event = threading.Event()
    
    def handle_keyboard_interrupt(signum, frame):
        print("\n🛑 Cancel requested by user...")
        cancel_event.set()
        progress_tracker.cancel()
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    
    try:
        ftp = ftplib.FTP(config["FTP_SERVER"])
        ftp.login(user=config["FTP_USER"], passwd=config["FTP_PASSWORD"])
        
        most_recent = find_most_recent_dir(ftp, config["REMOTE_DIR"])
        print(f"Exploring: {config['REMOTE_DIR']}/{most_recent}")
        print()
        
        # Count files before starting download to set correct total
        remote_path = f"{config['REMOTE_DIR']}/{most_recent}"
        total_files = count_files_recursively(ftp, remote_path, config)
        progress_tracker.start_operation(total_files)
        
        found = download_and_process_dir(
            ftp, remote_path, 
            config, progress_tracker, cancel_event
        )
        
        if found:
            print(f"\n✅ Download completed successfully!")
        else:
            print(f"\n⚠️ No files found matching criteria.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ftp_connection = locals().get('ftp')
        if ftp_connection:
            ftp_connection.quit()
        print(f"\n📊 Summary:")
        print(f"   Files downloaded: {progress_tracker.downloaded_files}")
        print(f"   Total files found: {progress_tracker.total_files}")
        if progress_tracker.downloaded_files > 0 and progress_tracker.start_time:
            elapsed = time.time() - progress_tracker.start_time
            print(f"   Time taken: {elapsed:.1f} seconds")


def main():
    """Main function with enhanced CLI support"""
    global FAILED_CHECKS
    FAILED_CHECKS = 0

    # Check for CLI mode
    if "--cli" in sys.argv:
        original_main_function()
        return
    elif len(sys.argv) > 1 and any(arg.startswith("--") for arg in sys.argv[1:]):
        # Other command line arguments - use original CLI mode
        original_main_function()
        return
    else:
        # Use enhanced CLI mode by default
        enhanced_cli_mode()


if __name__ == "__main__":
    main()
