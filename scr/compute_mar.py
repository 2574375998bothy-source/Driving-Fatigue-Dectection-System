
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# MediaPipe landmark indices for the mouth (these are standard)
# Using inner lip points for cleaner yawn signal
MOUTH_LANDMARKS = {
    "left_corner":  61,
    "right_corner": 291,
    "upper_lip":    13,    # inner upper lip center
    "lower_lip":    14,    # inner lower lip center
    "upper_outer":  0,     # outer upper lip (for visualization)
    "lower_outer":  17,    # outer lower lip (for visualization)
}


def compute_mar(landmarks, image_width, image_height):
    """
    Compute Mouth Aspect Ratio from face landmarks.

    Args:
        landmarks: list of MediaPipe landmark objects (normalized 0-1)
        image_width, image_height: pixel dimensions

    Returns:
        mar (float), and a dict of the (x, y) pixel coords used
    """
    def to_pixel(idx):
        lm = landmarks[idx]
        return np.array([lm.x * image_width, lm.y * image_height])

    left = to_pixel(MOUTH_LANDMARKS["left_corner"])
    right = to_pixel(MOUTH_LANDMARKS["right_corner"])
    upper = to_pixel(MOUTH_LANDMARKS["upper_lip"])
    lower = to_pixel(MOUTH_LANDMARKS["lower_lip"])

    horizontal_distance = np.linalg.norm(right - left)
    vertical_distance = np.linalg.norm(upper - lower)

    mar = vertical_distance / horizontal_distance if horizontal_distance > 0 else 0.0

    points = {
        "left": tuple(left.astype(int)),
        "right": tuple(right.astype(int)),
        "upper": tuple(upper.astype(int)),
        "lower": tuple(lower.astype(int)),
    }
    return mar, points


def process_image(image_path):
    """Load an image, run MediaPipe, compute MAR, return annotated image."""
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
    mar, points = compute_mar(landmarks, w, h)

    # Draw the 4 points used and the lines between them
    annotated = img.copy()
    cv2.line(annotated, points["left"], points["right"], (0, 255, 0), 2)
    cv2.line(annotated, points["upper"], points["lower"], (0, 0, 255), 2)
    for name, pt in points.items():
        cv2.circle(annotated, pt, 4, (255, 255, 0), -1)

    # Put MAR value on the image
    cv2.putText(annotated, f"MAR: {mar:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return annotated, mar


def main():
    yawn_path = next(Path("D:/Fatigue_project/data/dataset_new/train/yawn").glob("*.jpg"))
    no_yawn_path = next(Path("D:/Fatigue_project/data/dataset_new/train/no_yawn").glob("*.jpg"))

    print(f"Yawn sample:    {yawn_path.name}")
    yawn_result = process_image(yawn_path)
    if yawn_result is None:
        return
    yawn_img, yawn_mar = yawn_result
    print(f"  → MAR = {yawn_mar:.3f}")

    print(f"No-yawn sample: {no_yawn_path.name}")
    noyawn_result = process_image(no_yawn_path)
    if noyawn_result is None:
        return
    noyawn_img, noyawn_mar = noyawn_result
    print(f"  → MAR = {noyawn_mar:.3f}")

    # Sanity check: yawn should have higher MAR than no_yawn
    print(f"\n{'✓' if yawn_mar > noyawn_mar else '⚠'} "
          f"Yawn MAR ({yawn_mar:.3f}) "
          f"{'>' if yawn_mar > noyawn_mar else '<='} "
          f"No-yawn MAR ({noyawn_mar:.3f})")

    cv2.imshow("Yawn", yawn_img)
    cv2.imshow("No Yawn", noyawn_img)

    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
