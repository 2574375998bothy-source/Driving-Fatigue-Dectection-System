import os
from pathlib import Path

# Point to your data folder
DATA_DIR = Path("D:/Fatigue_project/data")

print(f"Looking inside: {DATA_DIR}\n")

# Walk through the data folder and show structure
for root, dirs, files in os.walk(DATA_DIR):
    # Calculate depth for indentation
    depth = root.replace(str(DATA_DIR), "").count(os.sep)
    indent = "  " * depth
    folder_name = os.path.basename(root)

    # Count image files in this folder
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if image_files:
        print(f"{indent}{folder_name}/  ({len(image_files)} images)")
    else:
        print(f"{indent}{folder_name}/")

    # Stop going too deep
    if depth >= 4:
        dirs.clear()

