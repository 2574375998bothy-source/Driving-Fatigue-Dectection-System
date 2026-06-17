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

# ============================================================
# 3D head model — generic average human face landmarks in mm.
# These are the "canonical" positions; solvePnP rotates this
# 3D model to match the 2D landmarks it sees in the image.
# ============================================================
FACE_3D_MODEL = np.array([
    (0.0,    0.0,    0.0),       # 1   Nose tip
    (0.0,    -330.0, -65.0),     # 152 Chin
    (-225.0,  170.0, -135.0),    # 33  Left eye outer corner
    (225.0,   170.0, -135.0),    # 263 Right eye outer corner
    (-150.0, -150.0, -125.0),    # 61  Left mouth corner
    (150.0,  -150.0, -125.0),    # 291 Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

DATA_DIR = Path("D:/Fatigue_project/Nodding_data")
OUTPUT_DIR = Path("D:/Fatigue_project/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_pitch_from_landmarks(landmarks, w, h):
    """
    Estimate head pitch angle (degrees) from MediaPipe landmarks
    using cv2.solvePnP.

    Returns:
        pitch in degrees, or None if PnP failed.
        Sign convention: depends on coords — script auto-detects later.
    """
    # Extract the 6 2D pixel positions
    image_points = np.array([
        (landmarks[idx].x * w, landmarks[idx].y * h)
        for idx in LANDMARK_INDICES
    ], dtype=np.float64)

    # Approximate camera matrix — no real calibration needed for
    # threshold-based detection. Using image width as focal length
    # is a standard webcam approximation.
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _tvec = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    # Convert rotation vector → rotation matrix → Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]  # X-axis rotation = pitch

    # solvePnP sometimes returns pitch near ±180°. Wrap into ±90°
    # so the values are intuitive ("head down = negative" etc.).
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return pitch


def process_folder(folder_path, face_mesh):
    pitches = []
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

        pitch = compute_pitch_from_landmarks(
            results.multi_face_landmarks[0].landmark, w, h
        )
        if pitch is None:
            failed += 1
            continue
        pitches.append(pitch)

        # Print progress every 20 images
        if i % 20 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  [{i}/{total}]  successful={len(pitches)}  failed={failed}  "
                  f"speed={rate:.1f} img/s  eta={eta:.0f}s", flush=True)

    return pitches, failed


def main():
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        nod_pitches,    nod_failed    = process_folder(DATA_DIR / "nod",    face_mesh)
        no_nod_pitches, no_nod_failed = process_folder(DATA_DIR / "no_nod", face_mesh)

    nod_pitches    = np.array(nod_pitches)
    no_nod_pitches = np.array(no_nod_pitches)

    print("\n=== Results ===")
    print(f"Nod     : {len(nod_pitches)} processed, {nod_failed} failed")
    print(f"          mean={nod_pitches.mean():.2f}°  std={nod_pitches.std():.2f}°  "
          f"min={nod_pitches.min():.2f}°  max={nod_pitches.max():.2f}°")
    print(f"No-nod  : {len(no_nod_pitches)} processed, {no_nod_failed} failed")
    print(f"          mean={no_nod_pitches.mean():.2f}°  std={no_nod_pitches.std():.2f}°  "
          f"min={no_nod_pitches.min():.2f}°  max={no_nod_pitches.max():.2f}°")

    # Auto-detect direction: is nodding the lower-pitch group or higher?
    nod_is_lower = nod_pitches.mean() < no_nod_pitches.mean()
    operator = "<=" if nod_is_lower else ">="
    print(f"\n📐 Detected rule: nod if pitch {operator} threshold")

    # Find best threshold by scanning the full range
    all_pitches = np.concatenate([nod_pitches, no_nod_pitches])
    thresholds  = np.linspace(all_pitches.min(), all_pitches.max(), 200)
    best_acc, best_t = 0, 0
    for t in thresholds:
        if nod_is_lower:
            correct = (nod_pitches <= t).sum() + (no_nod_pitches > t).sum()
        else:
            correct = (nod_pitches >= t).sum() + (no_nod_pitches < t).sum()
        acc = correct / (len(nod_pitches) + len(no_nod_pitches))
        if acc > best_acc:
            best_acc, best_t = acc, t

    print(f"\n🎯 Best threshold: pitch {operator} {best_t:.2f}°")
    print(f"   Accuracy: {best_acc * 100:.1f}%")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(no_nod_pitches, bins=30, alpha=0.6,
             label=f"no_nod (n={len(no_nod_pitches)})", color="green")
    plt.hist(nod_pitches, bins=30, alpha=0.6,
             label=f"nod (n={len(nod_pitches)})", color="red")
    plt.axvline(best_t, color="black", linestyle="--", linewidth=2,
                label=f"threshold = {best_t:.2f}°")
    plt.xlabel("Pitch angle (degrees)")
    plt.ylabel("Image count")
    plt.title(f"Pitch distribution — best threshold accuracy {best_acc*100:.1f}%\n"
              f"Rule: nod if pitch {operator} {best_t:.2f}°")
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = OUTPUT_DIR / "pitch_distribution.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()

    
