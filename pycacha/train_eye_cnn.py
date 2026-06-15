
import os
# Suppress noisy TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("D:/Fatigue_project/data/dataset_new/train")
OUTPUT_DIR = Path("D:/Fatigue_project/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 64       # resize all images to 64x64 - small enough to train fast
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2   # 20% of training data used for validation
SEED = 42


# ============================================================
# STEP 1: Load and preprocess images
# ============================================================
def load_images(folder_path, label):
    """Load all images from a folder, resize, convert to grayscale, return as array."""
    images = []
    labels = []
    image_files = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.png"))
    print(f"Loading {len(image_files)} images from '{folder_path.name}/' (label={label})...")

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
        labels.append(label)

    return images, labels


print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

closed_imgs, closed_lbls = load_images(DATA_DIR / "Closed", label=0)
open_imgs, open_lbls = load_images(DATA_DIR / "Open", label=1)

X = np.array(closed_imgs + open_imgs, dtype=np.float32)
y = np.array(closed_lbls + open_lbls, dtype=np.int32)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Add channel dimension (CNNs expect shape: [batch, height, width, channels])
X = X[..., np.newaxis]

print(f"\nDataset shape: X={X.shape}, y={y.shape}")
print(f"Class balance: Closed={np.sum(y==0)}, Open={np.sum(y==1)}")


# ============================================================
# STEP 2: Train/validation split
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Train/validation split")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y
)
print(f"Train: {X_train.shape[0]} images")
print(f"Val:   {X_val.shape[0]} images")


# ============================================================
# STEP 3: Build the CNN
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Building CNN architecture")
print("=" * 60)

model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

    # Conv block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Conv block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Conv block 3
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Classifier head
    layers.Flatten(),
    layers.Dropout(0.5),                     # prevents overfitting
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax"),   # 2 classes: Closed, Open
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# STEP 4: Train
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Training")
print("=" * 60)

# Early stopping: stop if val_loss doesn't improve for 3 epochs
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)


# ============================================================
# STEP 5: Save the model
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Saving model")
print("=" * 60)

model_path = OUTPUT_DIR / "eye_classifier.keras"
model.save(model_path)
print(f"✓ Model saved to: {model_path}")


# ============================================================
# STEP 6: Plot training curves
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Plotting training curves")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss
axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "training_curves.png"
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
print(f"✓ Plot saved to: {plot_path}")
plt.show()

print("\n" + "=" * 60)
print("DONE — training complete.")
print("=" * 60)

