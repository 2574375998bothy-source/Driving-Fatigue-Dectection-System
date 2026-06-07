"""
utils/api_client.py
====================
REST API client — all backend communication goes through here.

Backend API Contract (share with backend team):
────────────────────────────────────────────────────────────────────────────────

POST /api/verify
  Request body (JSON):
    {
      "driver_name": "John Doe",
      "frame":       "<base64-encoded JPEG>"
    }
  Response (JSON):
    {
      "status":  "authorized" | "unauthorized" | "obstructed" | "error",
      "name":    "John Doe",         // optional, matched name
      "confidence": 0.97             // optional, match confidence 0–1
    }

POST /api/analyze
  Request body (JSON):
    {
      "frame": "<base64-encoded JPEG>"
    }
  Response (JSON):
    {
      "status":    "NORMAL" | "DROWSY" | "YAWNING" | "ALERT",
      "ear":       0.312,             // Eye Aspect Ratio (float)
      "mar":       0.421,             // Mouth Aspect Ratio (float)
      "head_pose": "normal" | "anomaly",
      "landmarks": [[x,y], ...]       // optional, facial landmark list
    }

GET /api/health
  Response (JSON):
    { "ok": true }

────────────────────────────────────────────────────────────────────────────────
If the backend is unreachable, all methods return a "demo" fallback dict
so the frontend keeps running in demo mode.
"""

import base64
import io
import time
import numpy as np
import cv2
import requests
from typing import Optional


class APIClient:
    """
    Handles all HTTP calls to the backend.

    Parameters
    ----------
    base_url : str
        Root URL of the backend, e.g. "http://localhost:5000"
    timeout  : float
        Request timeout in seconds (default 3 s)
    """

    def __init__(self, base_url: str, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._last_alert_time: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    def verify_identity(self, frame: np.ndarray, driver_name: str) -> dict:
        """
        Send a frame to /api/verify and return the parsed response dict.
        Falls back to demo-mode dict on any network error.
        """
        payload = {
            "driver_name": driver_name,
            "frame": self._encode_frame(frame),
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/verify",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"[APIClient] verify_identity error: {exc}")
            # Demo fallback — treated as authorized so the UI keeps working
            return {"status": "demo", "name": driver_name}

    # ──────────────────────────────────────────────────────────────────────────
    def analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Send a frame to /api/analyze and return the parsed response dict.
        Falls back to a demo dict (randomised values) on network error.
        """
        payload = {"frame": self._encode_frame(frame)}
        try:
            resp = requests.post(
                f"{self.base_url}/api/analyze",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"[APIClient] analyze_frame error: {exc}")
            return self._demo_analysis()

    # ──────────────────────────────────────────────────────────────────────────
    def check_health(self) -> bool:
        """Return True if backend /api/health responds OK."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/health",
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode an OpenCV BGR frame to a base64 JPEG string."""
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _demo_analysis() -> dict:
        """
        Return a plausible demo analysis result when backend is offline.
        Values gently oscillate to simulate real detection.
        """
        t = time.time()
        ear = 0.30 + 0.05 * abs((t % 6) - 3) / 3  # oscillates 0.25–0.35
        mar = 0.35 + 0.10 * ((t % 10) / 10)        # slowly increases
        return {
            "status":    "NORMAL",
            "ear":       round(ear, 3),
            "mar":       round(mar, 3),
            "head_pose": "normal",
        }
