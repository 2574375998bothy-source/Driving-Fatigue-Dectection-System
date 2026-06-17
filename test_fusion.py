"""
Phase 6: Test the FatigueScorer logic without a webcam.
Simulates 4 scenarios and prints expected outcomes.
Run this first — if all 4 pass, your fusion logic is correct.

Usage:
    python D:\Fatigue_project\src\test_fusion.py
"""

from fatigue_scorer import FatigueScorer


def simulate(scorer, frames: list, label: str):
    """Feed a list of (eye_closed, is_yawning, is_nodding) tuples and report."""
    scorer.reset()
    # Reset internal counter manually for clean test
    scorer._bad_frame_count = 0

    final_result = None
    alert_frame  = None

    for i, (eye, yawn, nod) in enumerate(frames):
        result = scorer.update(eye, yawn, nod)
        final_result = result
        if result["alert"] and alert_frame is None:
            alert_frame = i + 1

    print(f"\n{'─'*55}")
    print(f"Scenario: {label}")
    print(f"  Total frames fed : {len(frames)}")
    print(f"  Final score      : {final_result['score']}")
    print(f"  Bad frame count  : {final_result['bad_frames']}")
    print(f"  Level            : {final_result['level']}")
    print(f"  Alert triggered  : {final_result['alert']}")
    if alert_frame:
        print(f"  Alert fired at frame #{alert_frame} (~{alert_frame/30:.1f}s)")


def main():
    scorer = FatigueScorer()

    print("=" * 55)
    print("Phase 6 — FatigueScorer Test Suite")
    print("=" * 55)

    # ── Scenario 1: Alert driver — all signals normal ──────────────────────────
    # 90 frames of nothing: score=0.0, no alert expected
    frames = [(False, False, False)] * 90
    simulate(scorer, frames, "1. Alert driver — no fatigue signals (expect: OK)")

    # ── Scenario 2: Eyes only closed for 1 second ─────────────────────────────
    # score = 0.4 (eye weight only) → above 0.5 threshold? No. Expect: OK
    frames = [(True, False, False)] * 30    # 1 second of eye closure
    simulate(scorer, frames, "2. Eyes closed 1s, nothing else (expect: OK — score 0.4 < 0.5)")

    # ── Scenario 3: Eyes + Yawning sustained for 3 seconds ────────────────────
    # score = 0.4 + 0.3 = 0.7 → above 0.5, sustained 90 frames → ALERT
    frames = [(True, True, False)] * 90
    simulate(scorer, frames, "3. Eyes closed + yawning for 3s (expect: ALERT at ~2s)")

    # ── Scenario 4: All three signals for 2 seconds ───────────────────────────
    # score = 0.4 + 0.3 + 0.3 = 1.0 → ALERT fires fast
    frames = [(True, True, True)] * 90
    simulate(scorer, frames, "4. All 3 signals active for 3s (expect: ALERT quickly)")

    # ── Scenario 5: Intermittent signals — should NOT alert ───────────────────
    # Alternating bad/good frames: bad_frame_count keeps resetting to 0
    frames = [(True, True, True), (False, False, False)] * 45   # 90 frames alternating
    simulate(scorer, frames, "5. Alternating good/bad frames (expect: OK — count resets)")

    print(f"\n{'='*55}")
    print("If scenarios 1, 2, 5 → OK  and  3, 4 → ALERT: ✅ PASS")
    print("=" * 55)


if __name__ == "__main__":
    main()

