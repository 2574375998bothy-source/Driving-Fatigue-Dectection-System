
import cv2
from pathlib import Path

# Project paths
DATA_DIR = Path("D:/Fatigue_project/data/dataset_new/train")
CLASSES = ["Closed", "Open", "yawn", "no_yawn"]

# Loop through each class folder
for class_name in CLASSES:
    class_dir = DATA_DIR / class_name

    # Get the first image in the folder
    image_files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))

    if not image_files:
        print(f"⚠  No images found in {class_dir}")
        continue

    first_image_path = image_files[0]
    print(f"Loading: {first_image_path.name} from class '{class_name}'")

    # Read image with OpenCV
    img = cv2.imread(str(first_image_path))

    if img is None:
        print(f"⚠  Failed to load {first_image_path}")
        continue

    # Print image info
    h, w = img.shape[:2]
    print(f"   Size: {w}x{h} pixels, Channels: {img.shape[2]}")

    # Display the image with class name as window title
    cv2.imshow(f"Class: {class_name}", img)

# Wait for any key press, then close all windows
print("\nPress any key in one of the image windows to close them all...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Done.")

