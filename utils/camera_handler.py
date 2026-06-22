"""
utils/camera_handler.py
========================
Wraps OpenCV VideoCapture with thread-safe frame access.
"""
from __future__ import annotations

import cv2
import sys
import threading
import time
import numpy as np


class CameraHandler:
    """
    Manages camera lifecycle and provides thread-safe frame access.

    Usage:
        cam = CameraHandler(index=0)
        cam.start()
        frame = cam.get_frame()   # returns np.ndarray or None
        cam.stop()
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Open camera and begin background capture loop."""
        if self._running:
            return
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend)
        if not self._cap.isOpened() and sys.platform == "win32":
            self._cap.release()
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            print(f"[CameraHandler] WARNING: Camera {self.camera_index} not available.")
            return
        # Keep only the newest frame and request a webcam-friendly format.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop capture loop and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None

    def get_frame(self) -> np.ndarray | None:
        """Return latest captured frame (BGR) or None if unavailable."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _capture_loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.02)
