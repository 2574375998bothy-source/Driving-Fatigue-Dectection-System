"""HTTP client used by the Tkinter frontend."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import cv2
import numpy as np
import requests


class APIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: str | None = None
        self.local_mode = os.environ.get("FATIGUE_LOCAL_BACKEND") == "1"
        self._engine = None
        self._engine_lock = threading.Lock()

    def _local_backend(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from detection_engine import DetectionEngine
                    project_root = Path(__file__).resolve().parent.parent
                    self._engine = DetectionEngine(project_root / "data" / "authorized_users")
        return self._engine

    @staticmethod
    def _jpeg(frame: np.ndarray) -> tuple[str, bytes]:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not encode camera frame")
        return "frame.jpg", encoded.tobytes()

    def check_health(self) -> bool:
        if self.local_mode:
            return True
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=2)
            return response.ok and response.json().get("status") == "ok"
        except (requests.RequestException, ValueError):
            return False

    def verify_identity(self, frame: np.ndarray, driver_name: str) -> dict:
        if self.local_mode:
            result = self._local_backend().verify(frame, driver_name)
            if result.get("status") == "authorized":
                self.session_id = result.get("session_id")
            return result
        try:
            response = requests.post(
                f"{self.base_url}/api/verify",
                files={"frame": self._jpeg(frame)},
                data={"driver_name": driver_name},
                timeout=self.timeout,
            )
            result = response.json()
            if response.ok and result.get("status") == "authorized":
                self.session_id = result.get("session_id")
            return result
        except requests.RequestException as exc:
            return {"status": "error", "message": f"Backend unavailable: {exc}"}
        except (ValueError, KeyError) as exc:
            return {"status": "error", "message": str(exc)}

    def enroll_identity(self, frames: list[np.ndarray], driver_name: str) -> dict:
        if self.local_mode:
            return self._local_backend().enroll(frames, driver_name)
        try:
            files = [("frames", self._jpeg(frame)) for frame in frames]
            response = requests.post(
                f"{self.base_url}/api/enroll",
                files=files,
                data={"driver_name": driver_name},
                timeout=max(self.timeout, 20.0),
            )
            return response.json()
        except requests.RequestException as exc:
            return {"status": "error", "message": f"Backend unavailable: {exc}"}
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

    def analyze_frame(self, frame: np.ndarray) -> dict:
        if not self.session_id:
            return {"status": "LOCKED", "message": "Driver has not been verified"}
        if self.local_mode:
            result, _ = self._local_backend().analyze(frame, self.session_id)
            return result
        try:
            response = requests.post(
                f"{self.base_url}/api/analyze",
                files={"frame": self._jpeg(frame)},
                data={"session_id": self.session_id},
                timeout=self.timeout,
            )
            return response.json()
        except requests.RequestException as exc:
            return {"status": "ERROR", "message": f"Backend unavailable: {exc}"}
        except ValueError as exc:
            return {"status": "ERROR", "message": str(exc)}

    def end_session(self) -> None:
        if self.local_mode:
            if self.session_id and self._engine is not None:
                self._engine.end_session(self.session_id)
            self.session_id = None
            return
        if self.session_id:
            try:
                requests.post(
                    f"{self.base_url}/api/session/end",
                    json={"session_id": self.session_id},
                    timeout=2,
                )
            except requests.RequestException:
                pass
        self.session_id = None
