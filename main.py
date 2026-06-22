"""Single-command desktop entry point with an in-process REST backend."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent


def backend_is_ready() -> bool:
    try:
        with urlopen("http://127.0.0.1:5000/api/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def prepare_import_paths() -> bool:
    """Map the project to an ASCII path required by MediaPipe on Windows."""
    root = PROJECT_ROOT
    mapped = False
    if os.name == "nt" and any(ord(char) > 127 for char in str(root)):
        subprocess.run(["subst", "R:", str(root)], capture_output=True, check=False)
        if Path("R:/server.py").exists():
            root = Path("R:/")
            mapped = True
    packages = root / ".venv" / "Lib" / "site-packages"
    for path in (packages, root, root / "scr"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return mapped


def relaunch_with_python311_if_needed() -> bool:
    """Allow `python main.py` even when `python` currently points to 3.14."""
    if sys.version_info[:2] == (3, 11):
        return False
    candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe"
    if not candidate.exists():
        raise RuntimeError("Python 3.11 is required. Install it and run main.py again.")
    mapped = prepare_import_paths()
    root = Path("R:/") if mapped else PROJECT_ROOT
    env = os.environ.copy()
    packages = root / ".venv" / "Lib" / "site-packages"
    if packages.exists():
        env["PYTHONPATH"] = str(packages)
    subprocess.run([str(candidate), str(root / "main.py")], cwd=str(root), env=env, check=False)
    if mapped:
        subprocess.run(["subst", "R:", "/D"], capture_output=True, check=False)
    return True


if __name__ == "__main__":
    if relaunch_with_python311_if_needed():
        raise SystemExit(0)
    mapped_drive = prepare_import_paths()
    os.environ["FATIGUE_LOCAL_BACKEND"] = "1"
    from scr.main_app import FatigueDetectionApp

    root = tk.Tk()
    app = FatigueDetectionApp(root)
    app.backend_mapped_drive = mapped_drive
    root.mainloop()
