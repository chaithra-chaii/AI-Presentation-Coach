"""
gestures.py - Hand gesture analysis for AI Presentation Coach (CV module).

Movement is measured in palm lengths per second, so it does not depend on how
far the speaker sits from the camera. Each hand is tracked separately.

Use in an app:
    from gestures import GestureAnalyzer

    analyzer = GestureAnalyzer()
    result = analyzer.analyze_frame(frame)               # frame = BGR image
    result = analyzer.analyze_frame(frame, face_box=box) # optional, see below
    print(analyzer.summary())
    analyzer.close()

face_box = (x_min, y_min, x_max, y_max) of the face in normalised 0..1
coordinates. If given, the analyzer reports when a hand is near the face.

Run this file directly to test with your webcam:
    python gestures.py
"""
import json
import math
import os
import statistics
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"
)

STATES = ("none", "still", "gesturing", "fidgeting")
LABELS = {
    "none": "Hands not visible",
    "still": "Hands still",
    "gesturing": "Gesturing - good",
    "fidgeting": "Fidgeting / too much movement",
}

MAX_GAP_SECONDS = 0.5        # bigger gaps between frames are not measured
WINDOW_SECONDS = 1.0         # movement is judged over the last second
MIN_WINDOW_SECONDS = 0.3     # need at least this much history to judge
NOISE_FLOOR = 0.012          # steps smaller than this (palm lengths) = camera noise
GRACE_SECONDS = 0.25         # hands lost for less than this still count as visible
POSITION_SMOOTHING = 0.6     # 0..1, higher reacts faster
FACE_MARGIN = 0.02           # extra space around the face box
TOUCH_POINTS = (4, 8, 12, 16, 20, 9)   # fingertips + palm centre


class _HandTrack:
    """Recent positions of one hand, in pixels."""

    def __init__(self):
        self.history = deque()
        self._sx = None
        self._sy = None

    def add(self, t, x, y, palm_px):
        if self._sx is None:
            self._sx, self._sy = x, y
        else:
            a = POSITION_SMOOTHING
            self._sx = a * x + (1 - a) * self._sx
            self._sy = a * y + (1 - a) * self._sy
        self.history.append((t, self._sx, self._sy, palm_px))
        while self.history and t - self.history[0][0] > WINDOW_SECONDS:
            self.history.popleft()

    def metrics(self):
        """Returns (speed in palm lengths/s, direction reversals per second)."""
        pts = list(self.history)
        if len(pts) < 3:
            return 0.0, 0.0
        duration = pts[-1][0] - pts[0][0]
        if duration < MIN_WINDOW_SECONDS:
            return 0.0, 0.0

        palm = statistics.median(p[3] for p in pts) or 1.0
        path, reversals, prev = 0.0, 0, None
        for a, b in zip(pts, pts[1:]):
            sx, sy = (b[1] - a[1]) / palm, (b[2] - a[2]) / palm
            mag = math.hypot(sx, sy)
            if mag < NOISE_FLOOR:
                continue
            path += mag
            if prev and (sx * prev[0] + sy * prev[1]) < 0:
                reversals += 1
            prev = (sx, sy)
        return path / duration, reversals / duration


class GestureAnalyzer:
    """Tracks hand visibility and movement quality across a session."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, still_limit=0.25,
                 fidget_speed=1.0, fidget_reversals=3.0, flail_speed=8.0,
                 num_hands=2):
        """
        Speeds are in palm lengths per second.
        still_limit      : below this = hands still
        fidget_speed     : minimum speed for fidgeting ...
        fidget_reversals : ... together with this many direction changes per second
        flail_speed      : faster than this is always counted as too much movement
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Hand model not found at: {model_path}\n"
                "Download hand_landmarker.task into the models folder."
            )
        self.still_limit = still_limit
        self.fidget_speed = fidget_speed
        self.fidget_reversals = fidget_reversals
        self.flail_speed = flail_speed

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # MediaPipe needs timestamps that always increase, so these are
        # never reset, even when a new session starts.
        self._origin = time.monotonic()
        self._last_ts = -1
        self.reset()

    def reset(self):
        """Start a new session (clears the summary)."""
        self._durations = {s: 0.0 for s in STATES}
        self._face_touch = 0.0
        self._tracks = {}
        self._prev_ts = None
        self._last_state = None
        self._last_seen = -1e9

    def _hand_state(self, speed, reversals):
        if speed < self.still_limit:
            return "still"
        if speed >= self.flail_speed or (
                speed >= self.fidget_speed and reversals >= self.fidget_reversals):
            return "fidgeting"
        return "gesturing"

    @staticmethod
    def _touching_face(hands, face_box):
        if not face_box:
            return False
        x0, y0, x1, y1 = face_box
        m = FACE_MARGIN
        for hand in hands:
            for j in TOUCH_POINTS:
                p = hand[j]
                if x0 - m <= p.x <= x1 + m and y0 - m <= p.y <= y1 + m:
                    return True
        return False

    def analyze_frame(self, frame_bgr, timestamp_ms=None, face_box=None):
        """
        Analyse one frame.
        Returns a dict: state, label, hands_count, speed, reversals_per_s,
        touching_face, landmarks
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("analyze_frame received an empty frame")

        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._origin) * 1000)
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = result.hand_landmarks or []
        handed = result.handedness or []

        dt = 0.0
        if self._prev_ts is not None:
            gap = (timestamp_ms - self._prev_ts) / 1000.0
            dt = min(gap, MAX_GAP_SECONDS)
            if gap > MAX_GAP_SECONDS:
                self._tracks.clear()
        self._prev_ts = timestamp_ms
        t = timestamp_ms / 1000.0

        speed, reversals, touching = 0.0, 0.0, False

        if hands:
            present, hand_states = set(), []
            for i, hand in enumerate(hands):
                key = (handed[i][0].category_name
                       if i < len(handed) and handed[i] else str(i))
                if key in present:
                    key = f"{key}_{i}"
                present.add(key)

                cx = sum(hand[j].x for j in (0, 5, 9, 13, 17)) / 5 * w
                cy = sum(hand[j].y for j in (0, 5, 9, 13, 17)) / 5 * h
                palm_px = math.hypot((hand[0].x - hand[9].x) * w,
                                     (hand[0].y - hand[9].y) * h)

                track = self._tracks.setdefault(key, _HandTrack())
                track.add(t, cx, cy, palm_px)
                sp, rev = track.metrics()
                speed, reversals = max(speed, sp), max(reversals, rev)
                hand_states.append(self._hand_state(sp, rev))

            for key in list(self._tracks):
                if key not in present:
                    del self._tracks[key]

            if "fidgeting" in hand_states:
                state = "fidgeting"
            elif "gesturing" in hand_states:
                state = "gesturing"
            else:
                state = "still"
            touching = self._touching_face(hands, face_box)
            self._last_state, self._last_seen = state, t
        elif self._last_state and t - self._last_seen <= GRACE_SECONDS:
            state = self._last_state            # brief detection drop: keep state
        else:
            state = "none"
            self._tracks.clear()
            self._last_state = None

        self._durations[state] += dt
        if touching:
            self._face_touch += dt

        label = LABELS[state]
        if touching:
            label += " - hand near face"

        return {
            "state": state,
            "label": label,
            "hands_count": len(hands),
            "speed": round(speed, 2),
            "reversals_per_s": round(reversals, 1),
            "touching_face": touching,
            "landmarks": [[(p.x, p.y) for p in hand] for hand in hands],
        }

    def summary(self):
        """Session results as percentages of time. Safe to call any time."""
        total = sum(self._durations.values())
        if total <= 0:
            return {"total_seconds": 0, "hands_visible_pct": 0,
                    "gesturing_pct": 0, "still_pct": 0, "fidgeting_pct": 0,
                    "face_touch_pct": 0}

        def pct(value):
            return round(value / total * 100, 1)

        return {
            "total_seconds": round(total, 1),
            "hands_visible_pct": round(100 - pct(self._durations["none"]), 1),
            "gesturing_pct": pct(self._durations["gesturing"]),
            "still_pct": pct(self._durations["still"]),
            "fidgeting_pct": pct(self._durations["fidgeting"]),
            "face_touch_pct": pct(self._face_touch),
        }

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_demo():
    """Webcam test with an on-screen display (not used by the app)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    colors = {"none": (0, 165, 255), "still": (255, 200, 0),
              "gesturing": (0, 200, 0), "fidgeting": (0, 0, 255)}

    with GestureAnalyzer() as analyzer:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = analyzer.analyze_frame(frame)
                h, w = frame.shape[:2]

                for hand in result["landmarks"]:
                    for x, y in hand:
                        cv2.circle(frame, (int(x * w), int(y * h)), 3,
                                   (0, 255, 0), -1)

                s = analyzer.summary()
                cv2.putText(frame, result["label"], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            colors[result["state"]], 2)
                cv2.putText(frame, f"Hands visible: {s['hands_visible_pct']:.0f}%",
                            (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"Gesturing: {s['gesturing_pct']:.0f}%   "
                                   f"Fidgeting: {s['fidgeting_pct']:.0f}%",
                            (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"speed: {result['speed']:.2f} palms/s   "
                                   f"direction changes: {result['reversals_per_s']:.1f}/s",
                            (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 0), 2)
                cv2.imshow("Gestures", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(json.dumps(analyzer.summary(), indent=2))


if __name__ == "__main__":
    run_demo()