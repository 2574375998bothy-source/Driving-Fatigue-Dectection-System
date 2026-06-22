"""Face authorization, obstruction detection and fatigue state machine."""

from __future__ import annotations

import threading
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from compute_mar import compute_mar
from fatigue_scorer import FatigueScorer
from nod_detector import compute_pitch, is_nodding
from occlusion_detector import is_camera_occluded

LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)


def eye_aspect_ratio(landmarks, indices, width: int, height: int) -> float:
    points = np.array(
        [(landmarks[i].x * width, landmarks[i].y * height) for i in indices],
        dtype=np.float64,
    )
    horizontal = np.linalg.norm(points[0] - points[3])
    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    return float(vertical / (2.0 * horizontal)) if horizontal else 0.0


@dataclass
class DriverSession:
    driver_name: str
    state: str = "MONITORING"
    scorer: FatigueScorer = field(default_factory=FatigueScorer)
    last_seen: float = field(default_factory=time.time)


class DetectionEngine:
    def __init__(self, authorized_dir: Path, ear_threshold=0.20, mar_threshold=0.24):
        self.authorized_dir = Path(authorized_dir)
        self.authorized_dir.mkdir(parents=True, exist_ok=True)
        self.ear_threshold = float(ear_threshold)
        self.mar_threshold = float(mar_threshold)
        self.sessions: dict[str, DriverSession] = {}
        self.lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.3
        )

    def _authorized_images(self) -> list[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        return [p for p in self.authorized_dir.iterdir() if p.suffix.lower() in extensions]

    @staticmethod
    def _driver_prefix(driver_name: str) -> str:
        digest = hashlib.sha256(driver_name.strip().encode("utf-8")).hexdigest()[:16]
        return f"driver_{digest}"

    @staticmethod
    def _read_image(path: Path) -> np.ndarray | None:
        try:
            return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, ValueError):
            return None

    def enroll(self, frames: list[np.ndarray], driver_name: str) -> dict:
        """Validate and store several reference samples for an authorized driver."""
        if not driver_name.strip():
            return {"status": "error", "message": "Driver name is required"}
        valid_frames = []
        for frame in frames:
            if frame is None or is_camera_occluded(frame):
                continue
            with self.processing_lock:
                detection = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if detection.detections:
                valid_frames.append(frame)
        if len(valid_frames) < 3:
            return {
                "status": "error",
                "message": "At least 3 clear face samples are required; improve lighting and face the camera",
            }

        prefix = self._driver_prefix(driver_name)
        # Replace this driver's old samples while retaining all other authorized drivers.
        for old_sample in self.authorized_dir.glob(f"{prefix}_*.jpg"):
            old_sample.unlink(missing_ok=True)
        timestamp = int(time.time() * 1000)
        saved = 0
        for index, frame in enumerate(valid_frames):
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                encoded.tofile(self.authorized_dir / f"{prefix}_{timestamp}_{index:02d}.jpg")
                saved += 1
        if saved < 3:
            return {"status": "error", "message": "Could not save enough face samples"}
        return {"status": "enrolled", "name": driver_name, "samples": saved}

    def _face_descriptor(self, image: np.ndarray) -> np.ndarray | None:
        """Small offline fallback when DeepFace weights are unavailable."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        else:
            with self.processing_lock:
                mesh_result = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not mesh_result.multi_face_landmarks:
                return None
            landmarks = mesh_result.multi_face_landmarks[0].landmark
            height, width = gray.shape
            xs = [point.x * width for point in landmarks]
            ys = [point.y * height for point in landmarks]
            x = max(0, int(min(xs)))
            y = max(0, int(min(ys)))
            w = min(width - x, max(1, int(max(xs)) - x))
            h = min(height - y, max(1, int(max(ys)) - y))
        face = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
        face = cv2.equalizeHist(face).astype(np.float32) / 255.0
        descriptor = cv2.dct(face)[:20, :20].flatten()[1:]
        norm = np.linalg.norm(descriptor)
        return descriptor / norm if norm else None

    def _offline_verify(self, frame: np.ndarray, references: list[Path]) -> tuple[bool, float]:
        query = self._face_descriptor(frame)
        if query is None:
            return False, 0.0
        best = 0.0
        for path in references:
            image = self._read_image(path)
            reference = self._face_descriptor(image) if image is not None else None
            if reference is not None:
                best = max(best, float(np.dot(query, reference)))
        return best >= 0.70, best

    def verify(self, frame: np.ndarray, claimed_name: str) -> dict:
        if is_camera_occluded(frame):
            return {"status": "obstructed", "message": "Camera is covered or image has no detail"}
        prefix = self._driver_prefix(claimed_name)
        named_references = list(self.authorized_dir.glob(f"{prefix}_*.jpg"))
        references = named_references or self._authorized_images()
        if not references:
            return {"status": "error", "message": "No authorized driver images are enrolled"}
        # Use the samples captured moments ago. This is local and deterministic;
        # it never waits for a large DeepFace model download during a request.
        authorized, similarity = self._offline_verify(frame, references)
        distance = 1.0 - similarity

        if not authorized:
            return {"status": "unauthorized", "message": "Face is not in the authorized database"}

        session_id = uuid.uuid4().hex
        with self.lock:
            self.sessions[session_id] = DriverSession(claimed_name or "Authorized driver")
        return {
            "status": "authorized",
            "name": claimed_name or "Authorized driver",
            "confidence": round(max(0.0, 1.0 - distance), 4),
            "session_id": session_id,
        }

    def analyze(self, frame: np.ndarray, session_id: str) -> tuple[dict, int]:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            return {"status": "LOCKED", "message": "Invalid or expired session"}, 401

        if is_camera_occluded(frame):
            session.state = "OCCLUDED"
            return {
                "status": "OCCLUDED", "state": session.state, "ear": 0.0, "mar": 0.0,
                "head_pose": "unknown", "message": "Camera obstruction detected",
            }, 200

        if session.state == "OCCLUDED":
            # An obstruction may hide a driver swap, so authorization is required again.
            self.end_session(session_id)
            return {"status": "LOCKED", "message": "Obstruction cleared; verify identity again"}, 401

        analysis_frame = frame
        if frame.shape[1] > 480:
            scale = 480.0 / frame.shape[1]
            analysis_frame = cv2.resize(
                frame, (480, max(1, int(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA
            )
        h, w = analysis_frame.shape[:2]
        with self.processing_lock:
            result = self.face_mesh.process(cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return {
                "status": "NO_FACE", "state": "MONITORING", "ear": 0.0, "mar": 0.0,
                "head_pose": "unknown", "message": "No face detected",
            }, 200

        landmarks = result.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar, _ = compute_mar(landmarks, w, h)
        pitch, _ = compute_pitch(landmarks, w, h)
        nodding = is_nodding(pitch)
        score = session.scorer.update(ear < self.ear_threshold, mar > self.mar_threshold, nodding)

        if score["level"] == "ALERT":
            status = "ALERT"
        elif score["level"] == "WARNING":
            status = "DROWSY"
        elif mar > self.mar_threshold:
            status = "YAWNING"
        else:
            status = "NORMAL"
        session.last_seen = time.time()
        return {
            "status": status,
            "state": "MONITORING",
            "ear": round(ear, 3),
            "mar": round(float(mar), 3),
            "pitch": round(float(pitch or 0.0), 2),
            "head_pose": "anomaly" if nodding else "normal",
            "fatigue_score": score["score"],
            "bad_frames": score["bad_frames"],
        }, 200

    def end_session(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)
