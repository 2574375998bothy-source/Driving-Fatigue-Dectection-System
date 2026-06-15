"""
Phase 2 - Debug version - find where it's silently failing
"""
print("[1] Script starting", flush=True)

import cv2
print("[2] cv2 imported", flush=True)

import mediapipe as mp
print("[3] mediapipe imported", flush=True)

from pathlib import Path
print("[4] pathlib imported", flush=True)

IMAGE_PATH = next(Path("D:/Fatigue_project/data/dataset_new/train/yawn").glob("*.jpg"))
print(f"[5] Found image: {IMAGE_PATH.name}", flush=True)

mp_face_mesh = mp.solutions.face_mesh
print("[6] mp_face_mesh assigned", flush=True)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
print("[7] FaceMesh initialized", flush=True)

img = cv2.imread(str(IMAGE_PATH))
print(f"[8] Image loaded, shape: {img.shape if img is not None else 'None'}", flush=True)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("[9] Converted to RGB", flush=True)

results = face_mesh.process(img_rgb)
print("[10] Face mesh processed", flush=True)

if results.multi_face_landmarks:
    print(f"[11] ✓ Detected {len(results.multi_face_landmarks)} face(s)", flush=True)
    print(f"[12] Landmarks: {len(results.multi_face_landmarks[0].landmark)}", flush=True)
else:
    print("[11] ⚠ No face detected", flush=True)

face_mesh.close()
print("[13] Done", flush=True)

