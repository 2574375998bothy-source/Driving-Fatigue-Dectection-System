"""
Nodding Detector — Phase 5
Computes head pitch angle from a face image / frame using cv2.solvePnP,
and returns whether the head is currently in a "nod" pose.

Designed to mirror compute_mar.py in shape and style so it plugs
into the integrated pipeline (Phase 6) the same way.
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# CONSTANTS
# ============================================================

# 3D head model — generic average human face landmark positions (mm).
# solvePnP rotates this 3D model to match the 2D landmarks it sees.
FACE_3D_MODEL = np.array([
    (0.0,    0.0,    0.0),       # Nose tip
    (0.0,    -330.0, -65.0),     # Chin
    (-225.0,  170.0, -135.0),    # Left eye outer corner
    (225.0,   170.0, -135.0),    # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0,  -150.0, -125.0),    # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Calibrated threshold from analyza_pitch.py (91.3% accuracy on 69 images)
PITCH_THRESHOLD = 13.0   # degrees — nod if pitch >= this value


# ============================================================
# CORE FUNCTIONS
# ============================================================

def compute_pitch(landmarks, image_width, image_height):
    """
    Compute head pitch angle (degrees) from MediaPipe face landmarks.

    Args:
        landmarks: list of MediaPipe landmark objects (normalized 0–1)
        image_width, image_height: pixel dimensions

    Returns:
        (pitch, points) where:
            pitch: float, degrees. None if PnP failed.
            points: dict of pixel (x, y) coords used for landmarks
                    (useful for visualization)
    """
    # Get the 6 2D pixel positions
    image_points = np.array([
        (landmarks[idx].x * image_width, landmarks[idx].y * image_height)
        for idx in LANDMARK_INDICES
    ], dtype=np.float64)

    # Approximate camera intrinsics — image width as focal length
    # is a standard webcam approximation; no calibration needed for
    # threshold-based detection.
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
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
        return None, None

    # Convert rotation vector → rotation matrix → Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]

    # Wrap pitch into ±90° so values are intuitive
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    points = {
        "nose":       tuple(image_points[0].astype(int)),
        "chin":       tuple(image_points[1].astype(int)),
        "left_eye":   tuple(image_points[2].astype(int)),
        "right_eye":  tuple(image_points[3].astype(int)),
        "left_mouth": tuple(image_points[4].astype(int)),
        "right_mouth": tuple(image_points[5].astype(int)),
    }
    return pitch, points


def is_nodding(pitch, threshold=PITCH_THRESHOLD):
    """
    Decide whether the head is currently in a nod pose.

    Args:
        pitch: float (degrees), or None
        threshold: float (degrees), default = calibrated PITCH_THRESHOLD

    Returns:
        True if head is tilted down past threshold, False otherwise.
        Treats pitch=None (PnP failure) as nodding — a face that
        disappears off the bottom of the frame is almost always
        an extreme head-down pose in the driver context.
    """
    if pitch is None:
        return True
    return pitch >= threshold


# ============================================================
# DEMO / SANITY CHECK
# ============================================================
def process_image(image_path):
    """Load an image, run MediaPipe, compute pitch, return annotated image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        print(f"⚠  No face detected in {image_path.name}")
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    pitch, points = compute_pitch(landmarks, w, h)

    if pitch is None:
        print(f"⚠  PnP failed for {image_path.name}")
        return None

    nodding = is_nodding(pitch)

    # Draw the landmarks used + an axis line through nose→chin
    annotated = img.copy()
    cv2.line(annotated, points["nose"], points["chin"], (0, 255, 0), 2)
    for name, pt in points.items():
        cv2.circle(annotated, pt, 4, (255, 255, 0), -1)

    # Overlay status text
    status_color = (0, 0, 255) if nodding else (0, 255, 0)
    status_text = "NODDING" if nodding else "NORMAL"
    cv2.putText(annotated, f"Pitch: {pitch:.1f}°", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(annotated, status_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    return annotated, pitch, nodding


def main():
    """Quick sanity check — one nod image and one no_nod image."""
    nod_path = next(Path("D:/Fatigue_project/Nodding_data/nod").glob("*.jpg"))
    no_nod_path = next(Path("D:/Fatigue_project/Nodding_data/no_nod").glob("*.jpg"))

    print(f"Threshold: pitch >= {PITCH_THRESHOLD}°  → NODDING\n")

    print(f"Nod sample:    {nod_path.name}")
    result = process_image(nod_path)
    if result is not None:
        nod_img, nod_pitch, nod_flag = result
        print(f"  → pitch = {nod_pitch:.2f}°  | is_nodding = {nod_flag}")
        cv2.imshow("Nod sample", nod_img)

    print(f"\nNo-nod sample: {no_nod_path.name}")
    result = process_image(no_nod_path)
    if result is not None:
        no_nod_img, no_nod_pitch, no_nod_flag = result
        print(f"  → pitch = {no_nod_pitch:.2f}°  | is_nodding = {no_nod_flag}")
        cv2.imshow("No-nod sample", no_nod_img)

    print("\nPress any key on an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
