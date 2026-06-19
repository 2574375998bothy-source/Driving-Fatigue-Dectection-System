import tkinter as tk
import sys
from pathlib import Path

# Add 'scr' to sys.path so it finds backend scripts properly
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scr"))

from scr.main_app import FatigueDetectionApp

if __name__ == "__main__":
    root = tk.Tk()
    app = FatigueDetectionApp(root)
    root.mainloop()
