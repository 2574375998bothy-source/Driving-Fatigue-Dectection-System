"""
utils/alert_manager.py
=======================
Manages alert sounds and cooldown logic.
Cross-platform: Windows (winsound) → macOS (afplay) → Linux (aplay/beep).
"""

import sys
import time
import threading
import subprocess


class AlertManager:
    """
    Plays audible alerts with per-type cooldown to avoid spam.

    Alert types:
        DROWSY       – repeated beeps (critical)
        YAWNING      – single beep (warning)
        HEAD         – double beep (warning)
        UNAUTHORIZED – long tone (security)
        OBSTRUCTION  – long tone (security)
    """

    COOLDOWNS = {
        "DROWSY":       2.5,
        "YAWNING":      4.0,
        "HEAD":         3.0,
        "UNAUTHORIZED": 5.0,
        "OBSTRUCTION":  5.0,
    }

    def __init__(self):
        self._last_alerts: dict[str, float] = {}
        self._lock = threading.Lock()

    def trigger(self, alert_type: str):
        """
        Fire an alert of the given type, respecting cooldown.
        Runs in a background thread so the UI never blocks.
        """
        now = time.time()
        cooldown = self.COOLDOWNS.get(alert_type, 3.0)
        with self._lock:
            last = self._last_alerts.get(alert_type, 0.0)
            if now - last < cooldown:
                return
            self._last_alerts[alert_type] = now

        threading.Thread(
            target=self._play, args=(alert_type,), daemon=True
        ).start()

    def _play(self, alert_type: str):
        """Play the appropriate sound for the alert type."""
        patterns = {
            "DROWSY":       [(800, 300), (800, 300), (800, 300)],  # 3 × beep
            "YAWNING":      [(600, 400)],                           # 1 × beep
            "HEAD":         [(700, 200), (700, 200)],               # 2 × beep
            "UNAUTHORIZED": [(400, 800)],                           # long low
            "OBSTRUCTION":  [(400, 800)],
        }
        sequence = patterns.get(alert_type, [(600, 300)])
        for freq, dur in sequence:
            self._beep(freq, dur)
            time.sleep(0.08)

    @staticmethod
    def _beep(freq: int, duration_ms: int):
        """Platform-agnostic beep."""
        if sys.platform == "win32":
            try:
                import winsound
                winsound.Beep(freq, duration_ms)
                return
            except Exception:
                pass

        if sys.platform == "darwin":  # macOS
            dur_s = duration_ms / 1000
            try:
                subprocess.run(
                    ["afplay", "-r", "1", "/System/Library/Sounds/Ping.aiff"],
                    timeout=dur_s + 0.5, capture_output=True
                )
                return
            except Exception:
                pass

        # Linux fallback
        try:
            subprocess.run(
                ["beep", f"-f{freq}", f"-l{duration_ms}"],
                timeout=2, capture_output=True
            )
        except Exception:
            # Last resort: print a bell character
            print("\a", end="", flush=True)
