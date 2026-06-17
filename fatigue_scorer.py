from collections import deque

EYE_WEIGHT   = 0.4
YAWN_WEIGHT  = 0.3
NOD_WEIGHT   = 0.3
ALERT_THRESHOLD   = 0.5
ALERT_FRAME_LIMIT = 60

class FatigueScorer:
    def __init__(self, eye_weight=EYE_WEIGHT, yawn_weight=YAWN_WEIGHT,
                 nod_weight=NOD_WEIGHT, alert_threshold=ALERT_THRESHOLD,
                 alert_frame_limit=ALERT_FRAME_LIMIT):
        self.eye_weight        = eye_weight
        self.yawn_weight       = yawn_weight
        self.nod_weight        = nod_weight
        self.alert_threshold   = alert_threshold
        self.alert_frame_limit = alert_frame_limit
        self._bad_frame_count  = 0
        self._history          = deque(maxlen=90)

    def update(self, eye_closed: bool, is_yawning: bool, is_nodding: bool) -> dict:
        score = (self.eye_weight  * int(eye_closed) +
                 self.yawn_weight * int(is_yawning) +
                 self.nod_weight  * int(is_nodding))
        if score > self.alert_threshold:
            self._bad_frame_count += 1
        else:
            self._bad_frame_count = 0
        self._history.append(score)
        alert = self._bad_frame_count >= self.alert_frame_limit
        if alert:
            level = "ALERT"
        elif self._bad_frame_count >= self.alert_frame_limit // 2:
            level = "WARNING"
        else:
            level = "OK"
        return {"score": round(score, 3), "bad_frames": self._bad_frame_count,
                "alert": alert, "level": level}

    def reset(self):
        self._bad_frame_count = 0

    @property
    def history(self):
        return list(self._history)
