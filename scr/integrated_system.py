"""Standalone Phase 8 demo using the same state machine as the REST backend."""

from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scr"))

from detection_engine import DetectionEngine

COLORS = {
    "NORMAL": (0, 200, 0), "YAWNING": (0, 180, 255), "DROWSY": (0, 100, 255),
    "ALERT": (0, 0, 255), "OCCLUDED": (0, 0, 255), "LOCKED": (0, 0, 255),
    "NO_FACE": (0, 180, 255),
}


def main() -> None:
    driver_name = input("Driver name: ").strip() or "Authorized driver"
    engine = DetectionEngine(PROJECT_ROOT / "data" / "authorized_users")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera 0 is unavailable")

    session_id = None
    frame_number = 0
    status = "LOCKED"
    message = "Look at the camera for authorization"

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame_number += 1

            if session_id is None and frame_number % 15 == 0:
                verification = engine.verify(frame, driver_name)
                status = verification["status"].upper()
                message = verification.get("message", "")
                if verification["status"] == "authorized":
                    session_id = verification["session_id"]
                    status, message = "NORMAL", "Authorized - fatigue monitor active"
            elif session_id is not None:
                result, _ = engine.analyze(frame, session_id)
                status = result["status"]
                message = result.get("message", "")
                if status == "LOCKED":
                    session_id = None

            color = COLORS.get(status, (255, 255, 255))
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (20, 20, 20), -1)
            cv2.putText(frame, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, message[:70], (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow("Integrated Driver Security and Fatigue Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if session_id:
            engine.end_session(session_id)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
