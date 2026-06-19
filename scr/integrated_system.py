"""
Phase 8: Integrated System with State Machine
Integrates Face Verification, Occlusion Detection, and Fatigue Monitoring.
"""

import cv2
import numpy as np
import mediapipe as mp
import keras
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scr")) # add scr too

from compute_mar import compute_mar
from nod_detector import is_nodding
from fatigue_scorer import FatigueScorer
from face_verification import verify_driver
from occlusion_detector import is_camera_occluded

# Parameters
MAR_THRESHOLD   = 0.240
PITCH_THRESHOLD = 13.03
EYE_IMG_SIZE    = (64, 64)

# States
STATE_AUTH = 0
STATE_MONITOR = 1
STATE_OCCLUDED = 2
STATE_LOCKED = 3

GREEN  = (0, 200, 0)
YELLOW = (0, 200, 255)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
FACE_6_IDX = [1, 152, 263, 33, 287, 57]

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0  ),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

def get_eye_crop(frame, landmarks, indices, h, w):
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
    x1, y1 = pts.min(axis=0) - 5
    x2, y2 = pts.max(axis=0) + 5
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, EYE_IMG_SIZE)

def predict_eye_closed(model, crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred = model.predict(img, verbose=0)
    return int(np.argmax(pred)) == 0

def get_pitch(landmarks, h, w):
    image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in FACE_6_IDX], dtype=np.float64)
    focal = w
    cam_matrix = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    success, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360

def draw_hud(frame, score_result, eye_closed, is_yawn, is_nod, pitch):
    level  = score_result["level"]
    score  = score_result["score"]
    bad_fr = score_result["bad_frames"]

    cv2.rectangle(frame, (0, 0), (340, 180), (20, 20, 20), -1)
    cv2.putText(frame, "FATIGUE MONITOR", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    def signal_row(label, active, y):
        colour = RED if active else GREEN
        status = "YES" if active else " NO"
        cv2.putText(frame, f"{label}: {status}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    signal_row("Eye Closed", eye_closed, 58)
    signal_row("Yawning   ", is_yawn,    82)
    signal_row(f"Nodding   ", is_nod,    106)
    cv2.putText(frame, f"Pitch: {pitch:+.1f} deg", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(frame, f"Score: {score:.2f}  ({bad_fr} bad frames)", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

    bar_w = int(score * 300)
    bar_colour = RED if score > 0.5 else GREEN
    cv2.rectangle(frame, (10, 162), (310, 175), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 162), (10 + bar_w, 175), bar_colour, -1)

    if level == "ALERT":
        cv2.rectangle(frame, (0, frame.shape[0]-60), (frame.shape[1], frame.shape[0]), RED, -1)
        cv2.putText(frame, "FATIGUE ALERT - Press R to reset", (30, frame.shape[0]-22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

def main():
    model_path = Path(r"D:\1\Intro\eye_classifier.keras")
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}.")
        return

    print("[INFO] Loading model...")
    eye_model = keras.models.load_model(str(model_path))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    scorer = FatigueScorer()
    cap = cv2.VideoCapture(0)

    state = STATE_AUTH
    auth_start_time = time.time()

    print("[INFO] Starting system...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        display_frame = frame.copy()

        if state == STATE_AUTH:
            cv2.putText(display_frame, "VERIFYING DRIVER...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, YELLOW, 2)
            
            # Check occlusion first
            if is_camera_occluded(frame):
                state = STATE_OCCLUDED
                continue

            if verify_driver(frame, str(PROJECT_ROOT / "data" / "authorized_users")):
                print("[INFO] Authorized user verified.")
                state = STATE_MONITOR
            else:
                if time.time() - auth_start_time > 10.0: # 10 seconds timeout
                    print("[INFO] Verification failed. Locked.")
                    state = STATE_LOCKED
                else:
                    cv2.putText(display_frame, "Face not recognized. Keep looking at camera.", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

        elif state == STATE_LOCKED:
            cv2.rectangle(display_frame, (0,0), (w,h), (0,0,255), -1)
            cv2.putText(display_frame, "SYSTEM LOCKED - UNAUTHORIZED", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

        elif state == STATE_OCCLUDED:
            cv2.rectangle(display_frame, (0,0), (w,h), (0,255,255), -1)
            cv2.putText(display_frame, "CAMERA OCCLUDED - ANTI-THEFT ALERT", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)
            # Check if occlusion is cleared
            if not is_camera_occluded(frame):
                print("[INFO] Occlusion cleared. Returning to auth.")
                state = STATE_AUTH
                auth_start_time = time.time()

        elif state == STATE_MONITOR:
            # First check for occlusion
            if is_camera_occluded(frame):
                state = STATE_OCCLUDED
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            
            eye_closed, is_yawn, is_nod, pitch = False, False, False, 0.0

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                for eye_idx in [LEFT_EYE, RIGHT_EYE]:
                    crop = get_eye_crop(frame, lm, eye_idx, h, w)
                    if crop is not None and predict_eye_closed(eye_model, crop):
                        eye_closed = True
                        break

                mar, _ = compute_mar(lm, w, h)
                is_yawn = mar > MAR_THRESHOLD
                pitch = get_pitch(lm, h, w)
                is_nod = pitch >= PITCH_THRESHOLD

            score_result = scorer.update(eye_closed, is_yawn, is_nod)
            draw_hud(display_frame, score_result, eye_closed, is_yawn, is_nod, pitch)

        cv2.imshow("Integrated System", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r') and state == STATE_MONITOR:
            scorer.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
