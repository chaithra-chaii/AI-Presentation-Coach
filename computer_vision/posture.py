import json
import math
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_lite.task"
)

STATES = ("good", "poor", "not_visible")
PROBLEMS = ("shoulders_uneven", "head_tilted", "slouching", "leaning_forward")
PROBLEM_LABELS = {
    "shoulders_uneven": "Shoulders uneven",
    "head_tilted": "Head tilted",
    "slouching": "Slouching",
    "leaning_forward": "Leaning forward",
}

# If two frames are further apart than this, the gap counts as this long
MAX_GAP_SECONDS = 0.5


def _angle_deg(p1, p2, w, h):
    """Tilt of the line between two points. 0 = perfectly level."""
    dx = (p2.x - p1.x) * w
    dy = (p2.y - p1.y) * h
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return ang


def _metrics(lm, w, h):
    """Returns shoulder_tilt, head_tilt, height_ratio, head_size."""
    l_sh, r_sh = lm[11], lm[12]
    l_ear, r_ear = lm[7], lm[8]

    shoulder_tilt = abs(_angle_deg(l_sh, r_sh, w, h))
    head_tilt = abs(_angle_deg(l_ear, r_ear, w, h))

    # Ear-to-shoulder height, relative to shoulder width
    sh_width = math.hypot((l_sh.x - r_sh.x) * w, (l_sh.y - r_sh.y) * h)
    sh_mid_y = (l_sh.y + r_sh.y) / 2 * h
    ear_mid_y = (l_ear.y + r_ear.y) / 2 * h
    height_ratio = (sh_mid_y - ear_mid_y) / sh_width if sh_width else 0.0

    # Head size on screen (ear to ear, in pixels): grows when leaning in
    head_size = math.hypot((l_ear.x - r_ear.x) * w, (l_ear.y - r_ear.y) * h)

    return shoulder_tilt, head_tilt, height_ratio, head_size


def _shoulders_visible(lm):
    return min((lm[11].visibility or 0), (lm[12].visibility or 0)) > 0.5


class PostureAnalyzer:
    """Tracks posture quality across a session."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, shoulder_tilt_limit=8,
                 head_tilt_limit=10, slouch_ratio=0.90, lean_ratio=1.12):
        """
        shoulder_tilt_limit : degrees, shoulders uneven above this
        head_tilt_limit     : degrees, head leaning sideways above this
        slouch_ratio        : below this share of upright height = slouching
        lean_ratio          : above this share of upright head size = leaning in
        Slouching and leaning need calibrate() to be called first.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Pose model not found at: {model_path}\n"
                "Download pose_landmarker_lite.task into the models folder."
            )
        self.shoulder_tilt_limit = shoulder_tilt_limit
        self.head_tilt_limit = head_tilt_limit
        self.slouch_ratio = slouch_ratio
        self.lean_ratio = lean_ratio

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        # MediaPipe needs timestamps that always increase, so these are
        # never reset, even when a new session starts.
        self._origin = time.monotonic()
        self._last_ts = -1
        self.baseline_height = None
        self.baseline_head_size = None
        self.reset()

    def reset(self):
        """Start a new session (clears the summary, keeps the calibration)."""
        self._durations = {s: 0.0 for s in STATES}
        self._problem_durations = {p: 0.0 for p in PROBLEMS}
        self._prev_ts = None

    @property
    def calibrated(self):
        return self.baseline_height is not None

    def _detect(self, frame_bgr, timestamp_ms=None):
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("Received an empty frame")

        if timestamp_ms is None:
            timestamp_ms = int((time.monotonic() - self._origin) * 1000)
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms), timestamp_ms

    def calibrate(self, frame_bgr):
        """
        Call while the speaker sits upright, head and both shoulders in view.
        Returns True if the calibration was saved.
        """
        result, _ = self._detect(frame_bgr)
        if not result.pose_landmarks:
            return False
        lm = result.pose_landmarks[0]
        if not _shoulders_visible(lm):
            return False
        h, w = frame_bgr.shape[:2]
        _, _, height_ratio, head_size = _metrics(lm, w, h)
        self.baseline_height = height_ratio
        self.baseline_head_size = head_size
        return True

    def analyze_frame(self, frame_bgr, timestamp_ms=None):
        """
        Analyse one frame.
        Returns a dict: state, label, problems, calibrated, metrics, points
        points = normalised (x, y) of ears and shoulders, for drawing.
        """
        result, ts = self._detect(frame_bgr, timestamp_ms)
        h, w = frame_bgr.shape[:2]

        dt = 0.0
        if self._prev_ts is not None:
            dt = min((ts - self._prev_ts) / 1000.0, MAX_GAP_SECONDS)
        self._prev_ts = ts

        problems, metrics, points = [], None, None

        if not result.pose_landmarks:
            state, label = "not_visible", "No person"
        else:
            lm = result.pose_landmarks[0]
            if not _shoulders_visible(lm):
                state, label = "not_visible", "Show both shoulders"
            else:
                sh_tilt, head_tilt, height, head_size = _metrics(lm, w, h)
                metrics = {
                    "shoulder_tilt": round(sh_tilt, 1),
                    "head_tilt": round(head_tilt, 1),
                    "height_ratio": round(height, 3),
                    "head_size": round(head_size, 1),
                }
                points = {name: (lm[i].x, lm[i].y) for name, i in
                          (("l_ear", 7), ("r_ear", 8),
                           ("l_shoulder", 11), ("r_shoulder", 12))}

                if sh_tilt > self.shoulder_tilt_limit:
                    problems.append("shoulders_uneven")
                if head_tilt > self.head_tilt_limit:
                    problems.append("head_tilted")
                if self.calibrated:
                    if height < self.baseline_height * self.slouch_ratio:
                        problems.append("slouching")
                    if head_size > self.baseline_head_size * self.lean_ratio:
                        problems.append("leaning_forward")

                if problems:
                    state = "poor"
                    label = ", ".join(PROBLEM_LABELS[p] for p in problems)
                else:
                    state, label = "good", "Good posture"

        self._durations[state] += dt
        for p in problems:
            self._problem_durations[p] += dt

        return {
            "state": state,
            "label": label,
            "problems": problems,
            "calibrated": self.calibrated,
            "metrics": metrics,
            "points": points,
        }

    def summary(self):
        """Percentages count only the time the speaker was visible."""
        total = sum(self._durations.values())
        tracked = self._durations["good"] + self._durations["poor"]
        if total <= 0 or tracked <= 0:
            return {"total_seconds": round(total, 1), "good_posture_pct": 0,
                    "visible_pct": 0,
                    **{f"{p}_pct": 0 for p in PROBLEMS}}

        out = {
            "total_seconds": round(total, 1),
            "good_posture_pct": round(self._durations["good"] / tracked * 100, 1),
            "visible_pct": round(tracked / total * 100, 1),
        }
        for p in PROBLEMS:
            out[f"{p}_pct"] = round(self._problem_durations[p] / tracked * 100, 1)
        return out

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

    colors = {"good": (0, 200, 0), "poor": (0, 0, 255),
              "not_visible": (0, 165, 255)}

    with PostureAnalyzer() as analyzer:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = analyzer.analyze_frame(frame)
                s = analyzer.summary()
                h, w = frame.shape[:2]
                color = colors[result["state"]]

                pts = result["points"]
                if pts:
                    px = {k: (int(x * w), int(y * h)) for k, (x, y) in pts.items()}
                    cv2.line(frame, px["l_shoulder"], px["r_shoulder"], color, 2)
                    cv2.line(frame, px["l_ear"], px["r_ear"], color, 2)
                    for p in px.values():
                        cv2.circle(frame, p, 6, color, -1)

                cv2.putText(frame, result["label"], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f"Good posture: {s['good_posture_pct']:.0f}%",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255, 255), 2)
                m = result["metrics"]
                if m:
                    cv2.putText(frame,
                                f"sh tilt: {m['shoulder_tilt']:.0f}  "
                                f"head tilt: {m['head_tilt']:.0f}  "
                                f"height: {m['height_ratio']:.2f}  "
                                f"head size: {m['head_size']:.0f}",
                                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 2)
                if not analyzer.calibrated:
                    cv2.putText(frame, "Sit upright and press C to calibrate",
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                cv2.imshow("Posture", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c'):
                    if analyzer.calibrate(frame):
                        analyzer.reset()   # start the percentages fresh
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(json.dumps(analyzer.summary(), indent=2))


if __name__ == "__main__":
    run_demo()