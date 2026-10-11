import json
import math
import os
import time

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
    "fidgeting": "Too much movement",
}

# If two frames are further apart than this, movement between them is ignored
MAX_GAP_SECONDS = 0.5


class GestureAnalyzer:
    """Tracks hand visibility and movement speed across a session."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, still_limit=0.05,
                 fidget_limit=0.90, smoothing=0.3, num_hands=2):
        """
        still_limit  : speed below this = hands still   (screen widths / second)
        fidget_limit : speed above this = too much movement
        smoothing    : 0..1, higher reacts faster, lower is smoother
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Hand model not found at: {model_path}\n"
                "Download hand_landmarker.task into the models folder."
            )
        self.still_limit = still_limit
        self.fidget_limit = fidget_limit
        self.smoothing = smoothing

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
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
        self._prev_centroid = None
        self._prev_count = 0
        self._prev_ts = None
        self._speed = 0.0

    def analyze_frame(self, frame_bgr, timestamp_ms=None):
        """
        Analyse one frame.
        frame_bgr    : image as a numpy array (OpenCV format)
        timestamp_ms : time of the frame in milliseconds (optional)
        Returns a dict: state, label, hands_count, speed, landmarks
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("analyze_frame received an empty frame")

        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._origin) * 1000)
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = result.hand_landmarks or []

        # Time since the previous analysed frame
        dt = 0.0
        if self._prev_ts is not None:
            dt = (timestamp_ms - self._prev_ts) / 1000.0
            if dt > MAX_GAP_SECONDS:
                self._prev_centroid = None   # gap too big to measure movement
                dt = MAX_GAP_SECONDS
        self._prev_ts = timestamp_ms

        if not hands:
            self._prev_centroid, self._prev_count = None, 0
            self._speed = 0.0
            state = "none"
        else:
            cx = sum(h[0].x for h in hands) / len(hands)   # wrist positions
            cy = sum(h[0].y for h in hands) / len(hands)

            # Only measure speed if the same number of hands was seen before
            if (self._prev_centroid is not None
                    and len(hands) == self._prev_count and dt > 0):
                dist = math.hypot(cx - self._prev_centroid[0],
                                  cy - self._prev_centroid[1])
                inst = dist / dt
                self._speed = (self.smoothing * inst
                               + (1 - self.smoothing) * self._speed)

            self._prev_centroid, self._prev_count = (cx, cy), len(hands)

            if self._speed < self.still_limit:
                state = "still"
            elif self._speed < self.fidget_limit:
                state = "gesturing"
            else:
                state = "fidgeting"

        self._durations[state] += dt   # time-based, so frame rate doesn't matter

        return {
            "state": state,
            "label": LABELS[state],
            "hands_count": len(hands),
            "speed": round(self._speed, 3),
            "landmarks": [[(p.x, p.y) for p in hand] for hand in hands],
        }

    def summary(self):
        """Session results as percentages of time. Safe to call any time."""
        total = sum(self._durations.values())
        if total <= 0:
            return {"total_seconds": 0, "hands_visible_pct": 0,
                    "gesturing_pct": 0, "still_pct": 0, "fidgeting_pct": 0}

        def pct(key):
            return round(self._durations[key] / total * 100, 1)

        return {
            "total_seconds": round(total, 1),
            "hands_visible_pct": round(100 - pct("none"), 1),
            "gesturing_pct": pct("gesturing"),
            "still_pct": pct("still"),
            "fidgeting_pct": pct("fidgeting"),
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
                h, w, _ = frame.shape

                for hand in result["landmarks"]:
                    for x, y in hand:
                        cv2.circle(frame, (int(x * w), int(y * h)), 3,
                                   (0, 255, 0), -1)

                s = analyzer.summary()
                cv2.putText(frame, result["label"], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            colors[result["state"]], 2)
                cv2.putText(frame, f"Hands visible: {s['hands_visible_pct']:.0f}%",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"Gesturing: {s['gesturing_pct']:.0f}%",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                cv2.putText(frame, f"speed: {result['speed']:.2f}",
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
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

