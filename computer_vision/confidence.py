"""
confidence.py - Confidence estimate from VISIBLE CUES for AI Presentation Coach.

This is an estimate, not a diagnosis. A camera cannot read feelings; it can only
see cues that often go with nervousness (blinking a lot, darting gaze, a
restless head, pressed lips, tense brows, wide eyes). Everything is compared
with the speaker's own relaxed face (calibration) where possible.

It reuses the face data produced by EyeContactAnalyzer, so no second face
model is loaded.

Use in an app:
    from confidence import ConfidenceAnalyzer

    conf = ConfidenceAnalyzer()
    conf.begin_calibration()        # speaker relaxes, neutral face, ~3 seconds
    ... keep calling conf.update(...) ...
    conf.finish_calibration()       # True if it worked

    eye = eye_analyzer.analyze_frame(frame)
    result = conf.update(time.monotonic(), eye["blendshapes"], eye["landmarks"])
    print(conf.summary())

Run this file directly to test with your webcam:
    python confidence.py
"""
import json
import math
import statistics
import time
from collections import deque

# ----- timing -------------------------------------------------------------
MAX_GAP_SECONDS = 0.5
BLINK_WINDOW = 30.0          # blinks per minute are measured over this window
MIN_OBSERVED = 10.0          # need this much face time before judging blinking
SHIFT_WINDOW = 10.0          # gaze shifts per second are measured over this window
HEAD_WINDOW = 3.0            # head movement is measured over this window
GAZE_LOOKBACK = 0.3          # a gaze shift = iris moved this much within 0.3 s
GAZE_SHIFT = 0.12
SHIFT_REFRACTORY = 0.4
BLINK_THRESHOLD = 0.5
BLINK_REFRACTORY = 0.15

# ----- how cues are turned into a score ------------------------------------
WEIGHTS = {"blinking": 0.18, "gaze_darting": 0.22, "restless_head": 0.20,
           "lip_press": 0.15, "brow_tension": 0.15, "eyes_wide": 0.10}
TENSION_GAIN = 2.0           # higher = more sensitive to nervous cues
SMILE_RELIEF = 0.25          # how much a smile lowers the tension
CONFIDENT_FROM = 70
NEUTRAL_FROM = 45

CUE_LABELS = {
    "blinking": "frequent blinking",
    "gaze_darting": "gaze darting around",
    "restless_head": "restless head movement",
    "lip_press": "pressed lips",
    "brow_tension": "tense or raised brows",
    "eyes_wide": "wide eyes",
}
STATE_LABELS = {
    "confident": "Looks confident",
    "neutral": "Looks neutral",
    "nervous": "Shows nervous cues",
    "calibrating": "Calibrating - relax, neutral face",
    "no_face": "No face",
}
LEVELS = ("confident", "neutral", "nervous")


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _gaze(lm):
    """Horizontal iris position across both eyes (0.5 = centre)."""
    def ratio(iris, outer, inner):
        width = lm[inner][0] - lm[outer][0]
        return 0.5 if width == 0 else (lm[iris][0] - lm[outer][0]) / width
    return (ratio(468, 33, 133) + ratio(473, 362, 263)) / 2


class ConfidenceAnalyzer:
    """Estimates confident vs nervous from visible face and head cues."""

    def __init__(self):
        self.baseline = None
        self._calibrating = False
        self._samples = []
        self.reset()

    # ----- session control -------------------------------------------------
    def reset(self):
        """Start a new session (clears the summary, keeps the calibration)."""
        self._durations = {s: 0.0 for s in LEVELS}
        self._conf_time = 0.0
        self._cue_time = {k: 0.0 for k in CUE_LABELS}
        self._prev_t = None
        self._observed = 0.0
        self._clear_windows()
        self._blink_times = deque()
        self._shift_times = deque()
        self._last_blink = -1e9
        self._last_shift = -1e9
        self._ema = {}

    def _clear_windows(self):
        self._gaze_hist = deque()
        self._head_hist = deque()
        self._blink_closed = False

    @property
    def calibrated(self):
        return self.baseline is not None

    @property
    def calibrating(self):
        return self._calibrating

    def begin_calibration(self):
        """Speaker relaxes with a neutral face. Keep calling update()."""
        self._calibrating = True
        self._samples = []

    def finish_calibration(self, min_samples=10):
        """Save the relaxed-face baseline. Returns True if enough samples."""
        self._calibrating = False
        samples, self._samples = self._samples, []
        if len(samples) < min_samples:
            return False
        self.baseline = {k: statistics.median(s[k] for s in samples)
                         for k in samples[0]}
        self.reset()
        return True

    def _ema_update(self, key, value, dt, tau):
        prev = self._ema.get(key)
        if prev is None:
            self._ema[key] = value
        elif dt > 0:
            self._ema[key] = prev + (1 - math.exp(-dt / tau)) * (value - prev)
        return self._ema[key]

    # ----- analysis --------------------------------------------------------
    def update(self, t, blendshapes, landmarks):
        """
        Call once per frame.
        t           : time in seconds (e.g. time.monotonic())
        blendshapes : dict of face blendshape scores (None if no face)
        landmarks   : list of (x, y) normalised face points (None if no face)
        Returns a dict: state, label, confidence, cues, top_cues, calibrated
        """
        dt = 0.0
        if self._prev_t is not None:
            gap = max(t - self._prev_t, 0.0)
            dt = min(gap, MAX_GAP_SECONDS)
            if gap > MAX_GAP_SECONDS:
                self._clear_windows()
        self._prev_t = t

        if not blendshapes or not landmarks:
            self._clear_windows()
            return self._result("no_face", None, {}, [])

        b = blendshapes

        def avg(a, c):
            return (b.get(a, 0.0) + b.get(c, 0.0)) / 2

        raw = {
            "press": avg("mouthPressLeft", "mouthPressRight"),
            "brow_inner": b.get("browInnerUp", 0.0),
            "brow_furrow": avg("browDownLeft", "browDownRight"),
            "wide": avg("eyeWideLeft", "eyeWideRight"),
            "smile": avg("mouthSmileLeft", "mouthSmileRight"),
        }

        if self._calibrating:
            self._samples.append(raw)
            return self._result("calibrating", None, {}, [])

        # --- blinks ---
        blink = avg("eyeBlinkLeft", "eyeBlinkRight")
        if blink > BLINK_THRESHOLD:
            if not self._blink_closed and t - self._last_blink > BLINK_REFRACTORY:
                self._blink_times.append(t)
                self._last_blink = t
            self._blink_closed = True
        else:
            self._blink_closed = False
        while self._blink_times and t - self._blink_times[0] > BLINK_WINDOW:
            self._blink_times.popleft()
        self._observed += dt
        blink_rate = None
        if self._observed >= MIN_OBSERVED:
            span = min(BLINK_WINDOW, self._observed)
            blink_rate = len(self._blink_times) / span * 60

        # --- gaze shifts and head movement ---
        shift_rate, head_motion = 0.0, 0.0
        if len(landmarks) >= 478:
            g = _gaze(landmarks)
            self._gaze_hist.append((t, g))
            while self._gaze_hist and t - self._gaze_hist[0][0] > GAZE_LOOKBACK:
                self._gaze_hist.popleft()
            if (abs(g - self._gaze_hist[0][1]) > GAZE_SHIFT
                    and t - self._last_shift > SHIFT_REFRACTORY):
                self._shift_times.append(t)
                self._last_shift = t
            while self._shift_times and t - self._shift_times[0] > SHIFT_WINDOW:
                self._shift_times.popleft()
            if self._observed >= 3.0:
                shift_rate = len(self._shift_times) / min(SHIFT_WINDOW, self._observed)

            face_w = landmarks[454][0] - landmarks[234][0]
            if face_w > 0:
                self._head_hist.append((t, landmarks[1][0], landmarks[1][1], face_w))
                while self._head_hist and t - self._head_hist[0][0] > HEAD_WINDOW:
                    self._head_hist.popleft()
            if len(self._head_hist) >= 10:
                xs = [p[1] for p in self._head_hist]
                ys = [p[2] for p in self._head_hist]
                head_motion = (math.hypot(statistics.pstdev(xs),
                                          statistics.pstdev(ys))
                               / self._head_hist[-1][3])

        # --- smooth the face-expression values, compare with relaxed face ---
        s = {k: self._ema_update(k, v, dt, 1.0) for k, v in raw.items()}
        base = self.baseline or {k: 0.0 for k in raw}

        cues = {
            "blinking": (_clamp01((blink_rate - 20) / 15)
                         if blink_rate is not None else 0.0),
            "gaze_darting": _clamp01((shift_rate - 0.4) / 0.8),
            "restless_head": _clamp01((head_motion - 0.03) / 0.05),
            "lip_press": _clamp01((s["press"] - base["press"]) / 0.25),
            "brow_tension": _clamp01(max(
                (s["brow_inner"] - base["brow_inner"]) / 0.3,
                (s["brow_furrow"] - base["brow_furrow"]) / 0.3)),
            "eyes_wide": _clamp01((s["wide"] - base["wide"]) / 0.3),
        }
        smile_cue = _clamp01((s["smile"] - base["smile"]) / 0.35)

        nervous_index = sum(WEIGHTS[k] * cues[k] for k in WEIGHTS)
        tension = _clamp01(nervous_index * TENSION_GAIN)
        conf_raw = 100 * (1 - _clamp01(tension - SMILE_RELIEF * smile_cue))
        conf = self._ema_update("conf", conf_raw, dt, 1.5)

        if conf >= CONFIDENT_FROM:
            state = "confident"
        elif conf >= NEUTRAL_FROM:
            state = "neutral"
        else:
            state = "nervous"

        self._durations[state] += dt
        self._conf_time += conf * dt
        for k, v in cues.items():
            self._cue_time[k] += v * dt

        top = [CUE_LABELS[k] for k, v in sorted(
            cues.items(), key=lambda kv: kv[1], reverse=True) if v >= 0.3][:3]
        return self._result(state, conf, cues, top, blink_rate, smile_cue)

    def _result(self, state, conf, cues, top, blink_rate=None, smile_cue=0.0):
        return {
            "state": state,
            "label": STATE_LABELS[state],
            "confidence": None if conf is None else round(conf),
            "cues": {k: round(v, 2) for k, v in cues.items()},
            "top_cues": top,
            "blinks_per_min": None if blink_rate is None else round(blink_rate),
            "smile": round(smile_cue, 2),
            "calibrated": self.calibrated,
        }

    def summary(self):
        """Percentages of time, average score and the most common cues."""
        total = sum(self._durations.values())
        if total <= 0:
            return {"total_seconds": 0, "avg_confidence": 0, "confident_pct": 0,
                    "neutral_pct": 0, "nervous_pct": 0, "top_cues": []}
        cue_avg = {k: v / total for k, v in self._cue_time.items()}
        top = [CUE_LABELS[k] for k, v in sorted(
            cue_avg.items(), key=lambda kv: kv[1], reverse=True) if v >= 0.2][:3]
        return {
            "total_seconds": round(total, 1),
            "avg_confidence": round(self._conf_time / total, 1),
            "confident_pct": round(self._durations["confident"] / total * 100, 1),
            "neutral_pct": round(self._durations["neutral"] / total * 100, 1),
            "nervous_pct": round(self._durations["nervous"] / total * 100, 1),
            "top_cues": top,
        }


def run_demo():
    """Webcam test with an on-screen display (not used by the app)."""
    import cv2
    from eye_contact import EyeContactAnalyzer

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    colors = {"confident": (0, 200, 0), "neutral": (255, 200, 0),
              "nervous": (0, 0, 255), "calibrating": (0, 255, 255),
              "no_face": (0, 165, 255)}
    calib_end, message, message_until = 0.0, "", 0.0

    eye = EyeContactAnalyzer()
    conf = ConfidenceAnalyzer()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            now = time.monotonic()

            e = eye.analyze_frame(frame)
            r = conf.update(now, e.get("blendshapes"), e.get("landmarks"))

            if conf.calibrating and now >= calib_end:
                ok = conf.finish_calibration()
                message = ("Calibrated!" if ok else
                           "Calibration failed: keep your face visible")
                message_until = now + 3

            for x, y in (e.get("landmarks") or []):
                cv2.circle(frame, (int(x * w), int(y * h)), 1, (255, 255, 0), -1)

            color = colors[r["state"]]
            cv2.putText(frame, r["label"], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if r["confidence"] is not None:
                cv2.putText(frame, f"Confidence: {r['confidence']}", (20, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.rectangle(frame, (20, 90), (220, 106), (200, 200, 200), 1)
                cv2.rectangle(frame, (20, 90),
                              (20 + 2 * r["confidence"], 106), color, -1)
            y = 135
            for k, v in r["cues"].items():
                cv2.putText(frame, f"{CUE_LABELS[k]}: {v:.2f}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 20
            cv2.putText(frame, f"smile: {r['smile']:.2f}   blinks/min: {r['blinks_per_min']}",
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if not conf.calibrated and not conf.calibrating:
                cv2.putText(frame, "Relax, neutral face, press C (3 seconds)",
                            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
            if now < message_until:
                cv2.putText(frame, message, (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow("Confidence", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c') and not conf.calibrating:
                conf.begin_calibration()
                calib_end = time.monotonic() + 3.0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        eye.close()
        print(json.dumps(conf.summary(), indent=2))


if __name__ == "__main__":
    run_demo()