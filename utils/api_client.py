"""
utils/api_client.py
====================
REST API client (replaced with local backend integration)

This module replaces the HTTP calls with direct imports and execution of the backend
system (integrated_system.py, fatigue_scorer.py, face_verification.py, etc.)
so the frontend and backend run seamlessly in the same process.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time

# Add 'scr' to sys.path to import backend modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scr"))

try:
    from face_verification import verify_driver
    from compute_mar import compute_mar
    from nod_detector import is_nodding
    from fatigue_scorer import FatigueScorer
    from integrated_system import get_pitch, predict_eye_closed, get_eye_crop, FACE_6_IDX, LEFT_EYE, RIGHT_EYE, MODEL_POINTS, PITCH_THRESHOLD, MAR_THRESHOLD
    import keras
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    
    model_path = PROJECT_ROOT / "scr" / "eye_status_cnn.h5"
    if model_path.exists():
        eye_model = keras.models.load_model(str(model_path))
    else:
        print("[APIClient] WARNING: eye_status_cnn.h5 not found.")
        eye_model = None

    scorer = FatigueScorer()
    BACKEND_AVAILABLE = True
except Exception as e:
    print(f"[APIClient] Backend import error: {e}")
    BACKEND_AVAILABLE = False


class APIClient:
    """
    Handles local backend processing (replaces HTTP calls).
    """

    def __init__(self, base_url: str = "", timeout: float = 3.0):
        self._last_alert_time: float = 0.0

    def verify_identity(self, frame: np.ndarray, driver_name: str) -> dict:
        """
        Runs local face verification on the frame.
        """
        if not BACKEND_AVAILABLE:
            print("[APIClient] Backend unavailable, running in demo mode.")
            return {"status": "demo", "name": driver_name}
            
        try:
            # verify_driver expects a raw frame, returns boolean
            is_verified = verify_driver(frame)
            if is_verified:
                return {"status": "authorized", "name": driver_name, "confidence": 1.0}
            else:
                return {"status": "unauthorized", "name": "Unknown", "confidence": 0.0}
        except Exception as exc:
            print(f"[APIClient] verify_identity error: {exc}")
            return {"status": "error", "name": driver_name}

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Runs local fatigue detection (Mediapipe + CNN + FatigueScorer).
        """
        if not BACKEND_AVAILABLE:
            return self._demo_analysis()
            
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            
            eye_closed = False
            is_yawn = False
            is_nod = False
            pitch = 0.0
            ear = 0.35 # Default mock value
            mar = 0.0
            
            h, w, _ = frame.shape

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                
                # Check eyes
                for eye_idx in [LEFT_EYE, RIGHT_EYE]:
                    crop = get_eye_crop(frame, lm, eye_idx, h, w)
                    if crop is not None and eye_model is not None:
                        if predict_eye_closed(eye_model, crop):
                            eye_closed = True
                            ear = 0.15 # Mock lower EAR if closed
                            break
                
                # Check mouth
                mar_val, _ = compute_mar(lm, w, h)
                mar = float(mar_val)
                is_yawn = mar > MAR_THRESHOLD
                
                # Check head pose (pitch)
                pitch = float(get_pitch(lm, h, w))
                is_nod = pitch >= PITCH_THRESHOLD

            score_result = scorer.update(eye_closed, is_yawn, is_nod)
            
            level_mapping = {
                "OK": "NORMAL",
                "WARNING": "DROWSY", # Map Warning to Drowsy for frontend compat
                "ALERT": "ALERT"
            }
            mapped_status = level_mapping.get(score_result["level"], "NORMAL")
            if is_yawn and mapped_status == "NORMAL":
                mapped_status = "YAWNING"
                
            return {
                "status": mapped_status,
                "ear": round(ear, 3),
                "mar": round(mar, 3),
                "head_pose": "anomaly" if is_nod else "normal"
            }
        except Exception as exc:
            print(f"[APIClient] analyze_frame error: {exc}")
            return self._demo_analysis()

    def check_health(self) -> bool:
        return BACKEND_AVAILABLE

    @staticmethod
    def _demo_analysis() -> dict:
        t = time.time()
        ear = 0.30 + 0.05 * abs((t % 6) - 3) / 3
        mar = 0.35 + 0.10 * ((t % 10) / 10)
        return {
            "status":    "NORMAL",
            "ear":       round(ear, 3),
            "mar":       round(mar, 3),
            "head_pose": "normal",
        }
