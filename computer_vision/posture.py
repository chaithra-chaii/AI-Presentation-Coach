"""
posture.py - Precise posture analysis for AI Presentation Coach (CV module).

Compares the speaker with THEIR OWN upright pose (calibration), smooths camera
noise, and flags small changes: uneven shoulders, head tilt, slouching,
leaning toward the camera, and swaying sideways.

Gesture-aware: while the speaker's hands are active (gesturing or touching the
face) the shoulder and head-tilt checks are relaxed, hidden ears are ignored,
and calibration frames are skipped, because moving arms and hands disturb the
shoulder and ear points.

Use in an app:
    from posture import PostureAnalyzer

    analyzer = PostureAnalyzer()
    analyzer.begin_calibration()             # speaker sits upright, hands down
    ... keep calling analyze_frame(frame) for about 2 seconds ...
    analyzer.finish_calibration()            # True if it worked

    result = analyzer.analyze_frame(frame, hands_active=False)
    print(analyzer.summary())
    analyzer.close()

    analyzer.calibrate(frame)  # quick single-frame calibration (less precise)

Run this file directly to test with your webcam:
    python posture.py
"""
import json
import math
import os
import statistics
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _default_model_path():
    """Prefer the more accurate 'full' model, fall back to 'lite'."""
    for name in ("pose_landmarker_full.task", "pose_landmarker_lite.task"):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isfile(path):
            return path
    return os.path.join(MODELS_DIR, "pose_landmarker_full.task")


STATES = ("good", "poor", "not_visible")
PROBLEMS = ("shoulders_uneven", "head_tilted", "slouching",
            "leaning_forward", "swaying")
PROBLEM_LABELS = {
    "shoulders_uneven": "Shoulders uneven",
    "head_tilted": "Head tilted",
    "slouching": "Slouching",
    "leaning_forward": "Leaning forward",
    "swaying": "Swaying sideways",
}
# How much each problem lowers the 0-100 score
WEIGHTS = {"shoulders_uneven": 14, "head_tilted": 8, "slouching": 22,
           "leaning_forward": 14, "swaying": 8}

MAX_GAP_SECONDS = 0.5
EAR_VISIBILITY = 0.5

# Gesture handling
GESTURE_TOL_FACTOR = 1.7     # shoulder/head tolerance multiplier while gesturing
GESTURE_HOLD = 0.6           # seconds the relaxed mode lasts after hands stop
GESTURE_SMOOTHING = 0.15     # stronger smoothing while gesturing

# Largest believable change per frame (degrees); bigger jumps are clipped
MAX_STEP = {"sh_tilt": 5.0, "head_tilt": 6.0}


def _angle_deg(p1, p2, w, h):
    """Signed tilt of the line between two points. 0 = perfectly level."""
    dx = (p2.x - p1.x) * w
    dy = (p2.y - p1.y) * h
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return ang


def _raw_metrics(lm, w, h):
    l_sh, r_sh, l_ear, r_ear = lm[11], lm[12], lm[7], lm[8]
    sh_width = math.hypot((l_sh.x - r_sh.x) * w, (l_sh.y - r_sh.y) * h)
    sh_mid_y = (l_sh.y + r_sh.y) / 2 * h
    ear_mid_y = (l_ear.y + r_ear.y) / 2 * h
    return {
        "sh_tilt": _angle_deg(l_sh, r_sh, w, h),
        "head_tilt": _angle_deg(l_ear, r_ear, w, h),
        "height": (sh_mid_y - ear_mid_y) / sh_width if sh_width else 0.0,
        "head_size": math.hypot((l_ear.x - r_ear.x) * w,
                                (l_ear.y - r_ear.y) * h),
        "sh_width": sh_width,
        "mid_x": (l_sh.x + r_sh.x) / 2 * w,
    }


def _shoulders_visible(lm):
    return min((lm[11].visibility or 0), (lm[12].visibility or 0)) > 0.5


def _ears_visible(lm):
    return min((lm[7].visibility or 0), (lm[8].visibility or 0)) > EAR_VISIBILITY


class PostureAnalyzer:
    """Tracks posture quality across a session."""

    def __init__(self, model_path=None, shoulder_tol=4.0, head_tol=5.0,
                 slouch_tol=0.07, lean_tol=0.06, sway_tol=0.10,
                 smoothing=0.4):
        """
        Tolerances (smaller = stricter):
          shoulder_tol : degrees the shoulder line may differ from your upright pose
          head_tol     : degrees the head may tilt from your upright pose
          slouch_tol   : fraction your head may sink toward your shoulders (0.07 = 7%)
          lean_tol     : fraction your head/shoulders may grow on screen (leaning in)
          sway_tol     : sideways shift, in shoulder widths
        smoothing      : 0..1, higher reacts faster, lower is smoother
        """
        model_path = model_path or _default_model_path()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Pose model not found at: {model_path}\n"
                "Download pose_landmarker_full.task (or lite) into the models folder."
            )
        self.tol = {"shoulders_uneven": shoulder_tol, "head_tilted": head_tol,
                    "slouching": slouch_tol, "leaning_forward": lean_tol,
                    "swaying": sway_tol}
        self.smoothing = smoothing
        self.model_path = model_path

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        # MediaPipe needs timestamps that always increase, so these are
        # never reset, even when a new session starts.
        self._origin = time.monotonic()
        self._last_ts = -1
        self.baseline = None
        self._calibrating = False
        self._samples = []
        self.reset()

    # ----- session control -------------------------------------------------
    def reset(self):
        """Start a new session (clears the summary, keeps the calibration)."""
        self._durations = {s: 0.0 for s in STATES}
        self._problem_durations = {p: 0.0 for p in PROBLEMS}
        self._score_time = 0.0
        self._prev_ts = None
        self._smooth = None
        self._active = set()
        self._gesture_until = -1e9

    @property
    def calibrated(self):
        return self.baseline is not None

    @property
    def calibrating(self):
        return self._calibrating

    @property
    def calibration_samples(self):
        return len(self._samples)

    # ----- calibration -----------------------------------------------------
    def begin_calibration(self):
        """Start collecting samples. Keep calling analyze_frame for ~2 seconds."""
        self._calibrating = True
        self._samples = []

    def finish_calibration(self, min_samples=10):
        """Save the average upright pose. Returns True if enough samples."""
        self._calibrating = False
        samples, self._samples = self._samples, []
        if len(samples) < min_samples:
            return False
        base = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
        if base["sh_width"] <= 0 or base["head_size"] <= 0 or base["height"] <= 0:
            return False
        self.baseline = base
        self.reset()
        return True

    def calibrate(self, frame_bgr):
        """Quick single-frame calibration (less precise than the 2-second one)."""
        result, _ = self._detect(frame_bgr)
        if not result.pose_landmarks:
            return False
        lm = result.pose_landmarks[0]
        if not _shoulders_visible(lm):
            return False
        h, w = frame_bgr.shape[:2]
        base = _raw_metrics(lm, w, h)
        if base["sh_width"] <= 0 or base["head_size"] <= 0 or base["height"] <= 0:
            return False
        self.baseline = base
        self.reset()
        return True

    # ----- analysis --------------------------------------------------------
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

    def _smooth_metrics(self, m, alpha):
        """Clip impossible jumps, then smooth: removes jitter, keeps real changes."""
        if self._smooth is None:
            self._smooth = dict(m)
            return self._smooth
        for k, v in m.items():
            prev = self._smooth[k]
            step = MAX_STEP.get(k)
            if step is not None:
                v = max(prev - step, min(prev + step, v))
            self._smooth[k] = alpha * v + (1 - alpha) * prev
        return self._smooth

    def _evaluate(self, m, factor=1.0):
        """Returns (problems, deviations, score 0-100)."""
        b = self.baseline
        tol = dict(self.tol)
        tol["shoulders_uneven"] *= factor     # relaxed while gesturing
        tol["head_tilted"] *= factor

        devs = {
            "shoulders_uneven": abs(m["sh_tilt"] - (b["sh_tilt"] if b else 0.0)),
            "head_tilted": abs(m["head_tilt"] - (b["head_tilt"] if b else 0.0)),
            "slouching": 0.0, "leaning_forward": 0.0, "swaying": 0.0,
        }
        if b:
            devs["slouching"] = max(0.0, (b["height"] - m["height"]) / b["height"])
            devs["leaning_forward"] = max(
                0.0, m["head_size"] / b["head_size"] - 1,
                m["sh_width"] / b["sh_width"] - 1)
            devs["swaying"] = abs(m["mid_x"] - b["mid_x"]) / b["sh_width"]

        problems = []
        for name in PROBLEMS:
            # Hysteresis: once flagged, stay flagged until clearly back in range
            limit = tol[name] * (0.8 if name in self._active else 1.0)
            if devs[name] > limit:
                problems.append(name)
        self._active = set(problems)

        penalty = sum(WEIGHTS[n] * min(devs[n] / tol[n], 1.5) for n in PROBLEMS)
        return problems, devs, max(0.0, 100.0 - penalty)

    def analyze_frame(self, frame_bgr, timestamp_ms=None, hands_active=False):
        """
        Analyse one frame.
        hands_active : True while the speaker's hands are gesturing or touching
                       the face (the combined analyzer passes this in)
        Returns a dict: state, label, problems, calibrated, calibrating,
        gesture_mode, score (0-100), metrics, points
        """
        result, ts = self._detect(frame_bgr, timestamp_ms)
        h, w = frame_bgr.shape[:2]
        t = ts / 1000.0

        dt = 0.0
        if self._prev_ts is not None:
            gap = (ts - self._prev_ts) / 1000.0
            dt = min(gap, MAX_GAP_SECONDS)
            if gap > MAX_GAP_SECONDS:
                self._smooth = None
        self._prev_ts = ts

        if hands_active:
            self._gesture_until = t + GESTURE_HOLD
        gesturing = t <= self._gesture_until

        problems, metrics, points, score = [], None, None, None

        if not result.pose_landmarks:
            state, label = "not_visible", "No person"
            self._smooth = None
        else:
            lm = result.pose_landmarks[0]
            if not _shoulders_visible(lm):
                state, label = "not_visible", "Show both shoulders"
                self._smooth = None
            else:
                raw = _raw_metrics(lm, w, h)
                ears_ok = _ears_visible(lm)

                # A hand hiding an ear makes the ear points unreliable:
                # keep the last good ear-based values instead.
                if not ears_ok and self._smooth is not None:
                    for k in ("head_tilt", "height", "head_size"):
                        raw[k] = self._smooth[k]

                # Only calibrate on clean frames (no gesturing, ears visible)
                if self._calibrating and not gesturing and ears_ok:
                    self._samples.append(dict(raw))

                alpha = GESTURE_SMOOTHING if gesturing else self.smoothing
                m = self._smooth_metrics(raw, alpha)
                factor = GESTURE_TOL_FACTOR if gesturing else 1.0
                problems, devs, score = self._evaluate(m, factor)

                metrics = {
                    "shoulder_dev_deg": round(devs["shoulders_uneven"], 1),
                    "head_dev_deg": round(devs["head_tilted"], 1),
                    "slouch_pct": round(devs["slouching"] * 100, 1),
                    "lean_pct": round(devs["leaning_forward"] * 100, 1),
                    "sway_pct": round(devs["swaying"] * 100, 1),
                }
                points = {name: (lm[i].x, lm[i].y) for name, i in
                          (("l_ear", 7), ("r_ear", 8),
                           ("l_shoulder", 11), ("r_shoulder", 12))}

                if self._calibrating:
                    state = "good"
                    label = ("Calibrating - keep hands down and still"
                             if gesturing else
                             "Calibrating - sit upright, hold still")
                elif problems:
                    state = "poor"
                    label = ", ".join(PROBLEM_LABELS[p] for p in problems)
                else:
                    state, label = "good", "Straight and confident"

        if not self._calibrating:
            self._durations[state] += dt
            if score is not None:
                self._score_time += score * dt
            for p in problems:
                self._problem_durations[p] += dt

        return {
            "state": state,
            "label": label,
            "problems": problems,
            "calibrated": self.calibrated,
            "calibrating": self._calibrating,
            "gesture_mode": gesturing,
            "score": None if score is None else round(score),
            "metrics": metrics,
            "points": points,
        }

    def summary(self):
        """Percentages count only the time the speaker was visible."""
        total = sum(self._durations.values())
        tracked = self._durations["good"] + self._durations["poor"]
        if total <= 0 or tracked <= 0:
            return {"total_seconds": round(total, 1), "good_posture_pct": 0,
                    "avg_posture_score": 0, "visible_pct": 0,
                    **{f"{p}_pct": 0 for p in PROBLEMS}}

        out = {
            "total_seconds": round(total, 1),
            "good_posture_pct": round(self._durations["good"] / tracked * 100, 1),
            "avg_posture_score": round(self._score_time / tracked, 1),
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
    calib_end, message, message_until = 0.0, "", 0.0

    with PostureAnalyzer() as analyzer:
        print("Using model:", os.path.basename(analyzer.model_path))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = analyzer.analyze_frame(frame)
                s = analyzer.summary()
                h, w = frame.shape[:2]
                color = colors[result["state"]]
                now = time.monotonic()

                if analyzer.calibrating and now >= calib_end:
                    ok = analyzer.finish_calibration()
                    message = ("Calibrated!" if ok else
                               "Calibration failed: keep head and both shoulders visible")
                    message_until = now + 3

                pts = result["points"]
                if pts:
                    px = {k: (int(x * w), int(y * h)) for k, (x, y) in pts.items()}
                    cv2.line(frame, px["l_shoulder"], px["r_shoulder"], color, 2)
                    cv2.line(frame, px["l_ear"], px["r_ear"], color, 2)
                    for p in px.values():
                        cv2.circle(frame, p, 6, color, -1)

                cv2.putText(frame, result["label"], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                score = result["score"]
                cv2.putText(frame,
                            f"Score: {score if score is not None else '-'}   "
                            f"Good posture: {s['good_posture_pct']:.0f}%",
                            (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                m = result["metrics"]
                if m:
                    cv2.putText(frame,
                                f"shoulders {m['shoulder_dev_deg']}deg  head {m['head_dev_deg']}deg  "
                                f"slouch {m['slouch_pct']}%  lean {m['lean_pct']}%  sway {m['sway_pct']}%",
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
                if not analyzer.calibrated and not analyzer.calibrating:
                    cv2.putText(frame, "Sit upright, hands down, press C (2 seconds)",
                                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                if analyzer.calibrating:
                    left = max(0.0, calib_end - now)
                    cv2.putText(frame, f"Calibrating... hold still {left:.1f}s",
                                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                if now < message_until:
                    cv2.putText(frame, message, (20, 175),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.imshow("Posture", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c') and not analyzer.calibrating:
                    analyzer.begin_calibration()
                    calib_end = time.monotonic() + 2.0
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(json.dumps(analyzer.summary(), indent=2))


if __name__ == "__main__":
    run_demo()