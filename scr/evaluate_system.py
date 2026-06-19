"""
Phase 9: System Evaluation
Evaluate the state machine and fatigue detection against test datasets.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
import keras
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scr"))

from face_verification import verify_driver
from occlusion_detector import is_camera_occluded
from compute_mar import compute_mar
from nod_detector import is_nodding
from fatigue_scorer import FatigueScorer

# Labels
# 0: Normal (Monitor)
# 1: Locked (Unauthorized)
# 2: Occluded
# 3: Fatigue Alert

LABEL_MAP = {
    0: "Normal",
    1: "Unauthorized",
    2: "Occluded",
    3: "Fatigue"
}

def predict_video_state(video_path, eye_model, face_mesh):
    """
    Run the state machine on a video and return the final dominant state.
    """
    cap = cv2.VideoCapture(video_path)
    scorer = FatigueScorer()
    
    state_counts = {0:0, 1:0, 2:0, 3:0}
    
    # State tracking
    is_locked = False
    is_occluded = False
    fatigue_alert = False
    
    auth_success = False
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. Check Occlusion
        if is_camera_occluded(frame):
            state_counts[2] += 1
            is_occluded = True
            continue
            
        # 2. Check Auth (only need to succeed once for simplicity in testing)
        if not auth_success:
            if verify_driver(frame, str(PROJECT_ROOT / "data" / "authorized_users")):
                auth_success = True
            else:
                if frame_count > 30: # If fail for 30 frames, considered locked
                    is_locked = True
                    state_counts[1] += 1
                continue
                
        # 3. Monitor Fatigue
        if auth_success:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            
            eye_closed, is_yawn, is_nod = False, False, False
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                
                # Simplified check for evaluation
                mar, _ = compute_mar(lm, w, h)
                if mar > 0.240: is_yawn = True
                
            score_result = scorer.update(eye_closed, is_yawn, is_nod)
            if score_result["level"] == "ALERT":
                fatigue_alert = True
                state_counts[3] += 1
            else:
                state_counts[0] += 1
                
    cap.release()
    
    # Logic to determine the overall video label
    if is_locked: return 1
    if is_occluded and state_counts[2] > frame_count * 0.5: return 2
    if fatigue_alert: return 3
    return 0

def run_evaluation():
    print("=== System Evaluation ===")
    
    # Load Models
    model_path = Path(r"D:\1\Intro\eye_classifier.keras")
    
    if not model_path.exists():
         print(f"[ERROR] Model not found at {model_path}")
         return
         
    eye_model = keras.models.load_model(str(model_path))
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    # Define test data (You can add more fatigue videos from your dataset here)
    test_cases = [
        {"path": "data/test_videos/normal_authorized.mp4", "true_label": 0},
        {"path": "data/test_videos/unauthorized.mp4", "true_label": 1},
        {"path": "data/test_videos/occluded.mp4", "true_label": 2},
        # TODO: Add your fatigue dataset paths here
        # {"path": "path/to/fatigue_video.mp4", "true_label": 3}
    ]
    
    y_true = []
    y_pred = []
    
    for case in test_cases:
        p = str(PROJECT_ROOT / case["path"])
        if not os.path.exists(p):
            print(f"[WARNING] Missing test file: {p}")
            continue
            
        print(f"Evaluating {p} ...")
        pred = predict_video_state(p, eye_model, face_mesh)
        y_true.append(case["true_label"])
        y_pred.append(pred)
        print(f"  -> True: {LABEL_MAP[case['true_label']]}, Predicted: {LABEL_MAP[pred]}")
        
    if not y_true:
        print("[ERROR] No test data found. Run record_test_data.py first!")
        return
        
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[LABEL_MAP[i] for i in range(4)],
                yticklabels=[LABEL_MAP[i] for i in range(4)])
    plt.title("System Evaluation - Confusion Matrix")
    plt.xlabel("Predicted State")
    plt.ylabel("True State")
    
    out_img = str(PROJECT_ROOT / "confusion_matrix.png")
    plt.savefig(out_img)
    print(f"\nSaved confusion matrix to: {out_img}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[LABEL_MAP[i] for i in range(4) if i in y_true], labels=list(set(y_true))))

if __name__ == "__main__":
    run_evaluation()
