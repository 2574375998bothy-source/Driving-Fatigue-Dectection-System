"""Phase 9 scenario evaluation and confusion-matrix generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scr"))

from detection_engine import DetectionEngine

LABELS = ["Normal", "Unauthorized", "Occluded", "Fatigue"]
TEST_CASES = [
    ("normal_authorized.mp4", "Normal"),
    ("unauthorized.mp4", "Unauthorized"),
    ("occluded.mp4", "Occluded"),
    ("fatigue.mp4", "Fatigue"),
]


def classify_video(path: Path, engine: DetectionEngine) -> str:
    capture = cv2.VideoCapture(str(path))
    session_id = None
    counts = {label: 0 for label in LABELS}
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number % 5:
                continue
            if session_id is None:
                result = engine.verify(frame, "Evaluation driver")
                if result["status"] == "authorized":
                    session_id = result["session_id"]
                    counts["Normal"] += 1
                elif result["status"] == "obstructed":
                    counts["Occluded"] += 1
                else:
                    counts["Unauthorized"] += 1
                continue

            result, _ = engine.analyze(frame, session_id)
            state = result["status"]
            if state == "OCCLUDED":
                counts["Occluded"] += 1
            elif state in ("ALERT", "DROWSY", "YAWNING"):
                counts["Fatigue"] += 1
            elif state == "LOCKED":
                counts["Unauthorized"] += 1
                session_id = None
            else:
                counts["Normal"] += 1
    finally:
        capture.release()
        if session_id:
            engine.end_session(session_id)
    return max(counts, key=counts.get)


def run_evaluation() -> None:
    engine = DetectionEngine(PROJECT_ROOT / "data" / "authorized_users")
    video_dir = PROJECT_ROOT / "data" / "test_videos"
    output_dir = PROJECT_ROOT / "evaluation_results"
    output_dir.mkdir(exist_ok=True)

    true_labels, predicted_labels, details = [], [], []
    for filename, expected in TEST_CASES:
        path = video_dir / filename
        if not path.exists():
            print(f"[SKIP] Missing {path}")
            continue
        predicted = classify_video(path, engine)
        true_labels.append(expected)
        predicted_labels.append(predicted)
        details.append({"video": filename, "expected": expected, "predicted": predicted})
        print(f"{filename}: expected={expected}, predicted={predicted}")

    if not true_labels:
        raise RuntimeError("No evaluation videos were found")

    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Driver Monitoring System Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    report = classification_report(
        true_labels, predicted_labels, labels=LABELS, output_dict=True, zero_division=0
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps({"cases": details, "report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    run_evaluation()
