"""Compatibility launcher. The complete application now starts from main.py."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(ROOT / "main.py")], cwd=ROOT))
