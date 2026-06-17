"""
Phase 6: Live Fatigue Demo 鈥?webcam integration.
Combines eye CNN, MAR yawn detector, and pitch nod detector
into a real-time fatigue score display.

Controls:
    Q  鈥?quit
    R  鈥?reset alert (acknowledge and start fresh)

Requirements:
    - outputs/eye_classifier.keras  (Phase 3)
    - src/compute_mar.py            (Phase 4)
    - src/nod_detector.py           (Phase 5)
    - src/fatigue_scorer.py         (Phase 6)

Run from project root:
    python D:\Fatigue_project\src\live_fatigue_demo.py
"""

import cv2
import numpy as np
import mediapipe as mp
import keras
from pathlib import Path
import sys

# 鈹€鈹€ Path setup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compute_mar import compute_mar          # your Phase 4 module
from nod_detector import is_nodding         # your Phase 5 module
from fatigue_scorer import FatigueScorer    # Phase 6 (this phase)

# 鈹€鈹€ Constants 鈥?adjust to match your Phase 4 & 5 results 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
MAR_THRESHOLD   = 0.240   # from your Phase 4 plot
PITCH_THRESHOLD = 13.03   # from your Phase 5 analyzer (91.3% accuracy)
EYE_IMG_SIZE    = (64, 64)

# 鈹€鈹€ Colours (BGR) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
GREEN  = (0, 200, 0)
YELLOW = (0, 200, 255)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

# 鈹€鈹€ MediaPipe eye landmark indices (same as Phase 3) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# 鈹€鈹€ 3D model points for solvePnP (same as Phase 5) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0  ),   # nose tip
    (0.0,   -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0),  # left eye corner
    (225.0,  170.0, -135.0),  # right eye corner
    (-150.0,-150.0, -125.0),  # left mouth corner
    (150.0, -150.0, -125.0),  # right mouth corner
], dtype=np.float64)

FACE_6_IDX = [1, 152, 263, 33, 287, 57]   # MediaPipe indices for the 6 points


def get_eye_crop(frame, landmarks, indices, h, w):
    """Extract a tight crop around one eye, return resized for CNN."""
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h))
                    for i in indices])
    x1, y1 = pts.min(axis=0) - 5
    x2, y2 = pts.max(axis=0) + 5
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, EYE_IMG_SIZE)


def predict_eye_closed(model, crop):
    """Return True if eye is closed. Class 0 = Closed, Class 1 = Open."""
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))   # (1, 64, 64, 1)
    pred = model.predict(img, verbose=0)
    return int(np.argmax(pred)) == 0           # 0 = Closed


def get_pitch(landmarks, h, w):
    """Compute head pitch angle using solvePnP. Returns degrees."""
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in FACE_6_IDX],
        dtype=np.float64
    )
    focal = w
    cam_matrix = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1    ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0] * 360   # convert to degrees
    return pitch


def draw_hud(frame, score_result, eye_closed, is_yawn, is_nod, pitch):
    """Draw the fatigue HUD overlay on the frame."""
    level  = score_result["level"]
    score  = score_result["score"]
    bad_fr = score_result["bad_frames"]

    # Background panel
    cv2.rectangle(frame, (0, 0), (340, 180), (20, 20, 20), -1)

    # Title
    cv2.putText(frame, "FATIGUE MONITOR", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # Signal rows
    def signal_row(label, active, y):
        colour = RED if active else GREEN
        status = "YES" if active else " NO"
        cv2.putText(frame, f"{label}: {status}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    signal_row("Eye Closed", eye_closed, 58)
    signal_row("Yawning   ", is_yawn,    82)
    signal_row(f"Nodding   ", is_nod,    106)
    cv2.putText(frame, f"Pitch: {pitch:+.1f} deg", (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    # Score bar
    cv2.putText(frame, f"Score: {score:.2f}  ({bad_fr} bad frames)",
                (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

    bar_w = int(score * 300)
    bar_colour = RED if score > 0.5 else GREEN
    cv2.rectangle(frame, (10, 162), (310, 175), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 162), (10 + bar_w, 175), bar_colour, -1)

    # Alert banner
    if level == "ALERT":
        cv2.rectangle(frame, (0, frame.shape[0]-60), (frame.shape[1], frame.shape[0]),
                      RED, -1)
        cv2.putText(frame, "鈿? FATIGUE ALERT  鈿? Press R to reset",
                    (30, frame.shape[0]-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    elif level == "WARNING":
        cv2.rectangle(frame, (0, frame.shape[0]-45), (frame.shape[1], frame.shape[0]),
                      YELLOW, -1)
        cv2.putText(frame, "WARNING 鈥?Stay alert!",
                    (30, frame.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, BLACK, 2)


def main():
    # Load eye CNN model
    model_path = PROJECT_ROOT / "outputs" / "eye_classifier.keras"
    if not model_path.exists():
        print(f"[ERROR] eye_classifier.keras not found at {model_path}")
        return
    print("[INFO] Loading eye classifier...")
    eye_model = keras.models.load_model(str(model_path))

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    scorer = FatigueScorer()
    cap    = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Running 鈥?press Q to quit, R to reset alert.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        eye_closed = False
        is_yawn    = False
        is_nod     = False
        pitch      = 0.0

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            # 鈹€鈹€ Eye signal 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            for eye_idx in [LEFT_EYE, RIGHT_EYE]:
                crop = get_eye_crop(frame, lm, eye_idx, h, w)
                if crop is not None:
                    if predict_eye_closed(eye_model, crop):
                        eye_closed = True
                        break   # one eye closed is enough

            # 鈹€鈹€ Yawn signal 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            mar, _ = compute_mar(lm, w, h)
            is_yawn = mar > MAR_THRESHOLD

            # 鈹€鈹€ Nodding signal 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            pitch  = get_pitch(lm, h, w)
            is_nod = pitch >= PITCH_THRESHOLD

        # 鈹€鈹€ Fuse into fatigue score 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        score_result = scorer.update(eye_closed, is_yawn, is_nod)

        # 鈹€鈹€ Draw HUD 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        draw_hud(frame, score_result, eye_closed, is_yawn, is_nod, pitch)

        cv2.imshow("Driver Fatigue Monitor 鈥?Phase 6", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            scorer.reset()
            print("[INFO] Alert reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

