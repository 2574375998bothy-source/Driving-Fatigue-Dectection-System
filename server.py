"""REST backend for the Driver Fatigue Detection System."""

from pathlib import Path
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "scr"))

from detection_engine import DetectionEngine

app = Flask(__name__)
engine = DetectionEngine(PROJECT_ROOT / "data" / "authorized_users")


def uploaded_frame():
    upload = request.files.get("frame")
    if upload is None:
        return None
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def uploaded_frames():
    frames = []
    for upload in request.files.getlist("frames"):
        data = np.frombuffer(upload.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames


@app.get("/api/health")
def health():
    return jsonify(status="ok", service="fatigue-detection-backend")


@app.post("/api/verify")
def verify():
    frame = uploaded_frame()
    if frame is None:
        return jsonify(status="error", message="A valid JPEG/PNG frame is required"), 400
    result = engine.verify(frame, request.form.get("driver_name", "").strip())
    code = 200 if result["status"] != "error" else 503
    return jsonify(result), code


@app.post("/api/enroll")
def enroll():
    frames = uploaded_frames()
    if not frames:
        return jsonify(status="error", message="Face sample frames are required"), 400
    result = engine.enroll(frames, request.form.get("driver_name", "").strip())
    return jsonify(result), (200 if result["status"] == "enrolled" else 400)


@app.post("/api/analyze")
def analyze():
    frame = uploaded_frame()
    if frame is None:
        return jsonify(status="ERROR", message="A valid JPEG/PNG frame is required"), 400
    result, code = engine.analyze(frame, request.form.get("session_id", ""))
    return jsonify(result), code


@app.post("/api/session/end")
def end_session():
    payload = request.get_json(silent=True) or {}
    engine.end_session(payload.get("session_id", ""))
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
