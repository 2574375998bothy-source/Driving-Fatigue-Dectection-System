
import os
# Suppress noisy MediaPipe / TF logs BEFORE importing them
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

MOUTH_LANDMARKS = {
    "left_corner":  61,
    "right_corner": 291,
    "upper_lip":    13,
    "lower_lip":    14,
}

DATA_DIR = Path("D:/Fatigue_project/data/dataset_new/train")
OUTPUT_DIR = Path("D:/Fatigue_project/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_mar_from_landmarks(landmarks, w, h):
    def px(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    horizontal = np.linalg.norm(px(MOUTH_LANDMARKS["right_corner"]) - px(MOUTH_LANDMARKS["left_corner"]))
    vertical = np.linalg.norm(px(MOUTH_LANDMARKS["upper_lip"]) - px(MOUTH_LANDMARKS["lower_lip"]))
    return vertical / horizontal if horizontal > 0 else 0.0


def process_folder(folder_path, face_mesh):
    mar_values = []
    failed = 0
    image_files = sorted(folder_path.glob("*.jpg")) + sorted(folder_path.glob("*.png"))
    total = len(image_files)
    print(f"\nProcessing {total} images from '{folder_path.name}/'...", flush=True)

    start = time.time()
    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            failed += 1
            continue
        mar = compute_mar_from_landmarks(results.multi_face_landmarks[0].landmark, w, h)
        mar_values.append(mar)

        # Print progress every 50 images
        if i % 50 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  [{i}/{total}]  successful={len(mar_values)}  failed={failed}  "
                  f"speed={rate:.1f} img/s  eta={eta:.0f}s", flush=True)

    return mar_values, failed


def main():
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        yawn_mars, yawn_failed = process_folder(DATA_DIR / "yawn", face_mesh)
        noyawn_mars, noyawn_failed = process_folder(DATA_DIR / "no_yawn", face_mesh)

    yawn_mars = np.array(yawn_mars)
    noyawn_mars = np.array(noyawn_mars)

    print("\n=== Results ===")
    print(f"Yawn    : {len(yawn_mars)} processed, {yawn_failed} failed")
    print(f"          mean={yawn_mars.mean():.3f}  std={yawn_mars.std():.3f}  "
          f"min={yawn_mars.min():.3f}  max={yawn_mars.max():.3f}")
    print(f"No-yawn : {len(noyawn_mars)} processed, {noyawn_failed} failed")
    print(f"          mean={noyawn_mars.mean():.3f}  std={noyawn_mars.std():.3f}  "
          f"min={noyawn_mars.min():.3f}  max={noyawn_mars.max():.3f}")

    # Find best threshold
    thresholds = np.linspace(0.0, 1.5, 151)
    best_acc, best_t = 0, 0
    for t in thresholds:
        correct = (yawn_mars >= t).sum() + (noyawn_mars < t).sum()
        acc = correct / (len(yawn_mars) + len(noyawn_mars))
        if acc > best_acc:
            best_acc, best_t = acc, t

    print(f"\n🎯 Best threshold: MAR >= {best_t:.3f}")
    print(f"   Accuracy: {best_acc * 100:.1f}%")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(noyawn_mars, bins=50, alpha=0.6, label=f"no_yawn (n={len(noyawn_mars)})", color="green")
    plt.hist(yawn_mars, bins=50, alpha=0.6, label=f"yawn (n={len(yawn_mars)})", color="red")
    plt.axvline(best_t, color="black", linestyle="--", linewidth=2, label=f"threshold = {best_t:.3f}")
    plt.xlabel("MAR")
    plt.ylabel("Image count")
    plt.title(f"MAR distribution — best threshold accuracy {best_acc*100:.1f}%")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = OUTPUT_DIR / "mar_distribution.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
