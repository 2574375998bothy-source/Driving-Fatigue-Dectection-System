
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
)


# ============================================================
# CONFIG
# ============================================================
TEST_DIR = Path("D:/Fatigue_project/data/dataset_new/test")
MODEL_PATH = Path("D:/Fatigue_project/outputs/eye_classifier.keras")
OUTPUT_DIR = Path("D:/Fatigue_project/outputs")

IMAGE_SIZE = 64
CLASS_NAMES = ["Closed", "Open"]   # label 0 = Closed, label 1 = Open


# ============================================================
# STEP 1: Load the model
# ============================================================
print("=" * 60)
print("STEP 1: Loading trained model")
print("=" * 60)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first!")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Loaded model from {MODEL_PATH}")


# ============================================================
# STEP 2: Load test images
# ============================================================
def load_images(folder_path, label):
    images, labels = [], []
    image_files = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.png"))
    print(f"Loading {len(image_files)} from '{folder_path.name}/' (label={label})...")
    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
        labels.append(label)
    return images, labels


print("\n" + "=" * 60)
print("STEP 2: Loading test data")
print("=" * 60)

closed_imgs, closed_lbls = load_images(TEST_DIR / "Closed", label=0)
open_imgs, open_lbls = load_images(TEST_DIR / "Open", label=1)

X_test = np.array(closed_imgs + open_imgs, dtype=np.float32) / 255.0
X_test = X_test[..., np.newaxis]                      # add channel dimension
y_test = np.array(closed_lbls + open_lbls, dtype=np.int32)

print(f"\nTest set: {X_test.shape[0]} images")
print(f"  Closed: {np.sum(y_test == 0)}")
print(f"  Open:   {np.sum(y_test == 1)}")


# ============================================================
# STEP 3: Predict
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Running predictions")
print("=" * 60)

probabilities = model.predict(X_test, batch_size=32, verbose=1)
y_pred = np.argmax(probabilities, axis=1)


# ============================================================
# STEP 4: Compute metrics
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Metrics")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))


# ============================================================
# STEP 5: Confusion matrix
# ============================================================
print("=" * 60)
print("STEP 5: Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print(f"\n{cm}\n")
print(f"        Predicted Closed   Predicted Open")
print(f"Actual Closed  {cm[0,0]:4d}              {cm[0,1]:4d}")
print(f"Actual Open    {cm[1,0]:4d}              {cm[1,1]:4d}")

# Plot it
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
ax.set_title(f"Eye Classifier — Test Set Confusion Matrix\nAccuracy: {accuracy*100:.2f}%")
plt.tight_layout()

cm_path = OUTPUT_DIR / "confusion_matrix_eyes.png"
plt.savefig(cm_path, dpi=120, bbox_inches="tight")
print(f"\n✓ Confusion matrix saved to: {cm_path}")
plt.show()


# ============================================================
# STEP 6: Show a few misclassified examples
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Misclassified examples")
print("=" * 60)

wrong_indices = np.where(y_pred != y_test)[0]
print(f"Total misclassified: {len(wrong_indices)} / {len(y_test)}")

if len(wrong_indices) > 0:
    n_show = min(6, len(wrong_indices))
    fig, axes = plt.subplots(1, n_show, figsize=(15, 3))
    if n_show == 1:
        axes = [axes]
    for ax, idx in zip(axes, wrong_indices[:n_show]):
        ax.imshow(X_test[idx].squeeze(), cmap="gray")
        ax.set_title(f"True: {CLASS_NAMES[y_test[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}",
                     fontsize=10, color="red")
        ax.axis("off")
    plt.tight_layout()
    miss_path = OUTPUT_DIR / "misclassified_eyes.png"
    plt.savefig(miss_path, dpi=120, bbox_inches="tight")
    print(f"✓ Examples saved to: {miss_path}")
    plt.show()

print("\n" + "=" * 60)
print("Evaluation complete.")
print("=" * 60)


