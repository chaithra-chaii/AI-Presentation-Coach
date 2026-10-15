"""
analyzer.py - Combined computer vision analyzer for AI Presentation Coach.

Runs eye contact, posture, gestures and a confidence estimate on the same
webcam frames.

Use in an app:
    from analyzer import CVAnalyzer

    cv_analyzer = CVAnalyzer()

    # Calibration: speaker sits upright, relaxed neutral face, looks at camera
    cv_analyzer.begin_calibration()
    ... keep calling analyze_frame(frame) for about 3 seconds ...
    cv_analyzer.finish_calibration(frame)    # {"ok": True, ...}

    result = cv_analyzer.analyze_frame(frame)   # frame = BGR image (numpy array)
    report = cv_analyzer.summary()              # one dict for the whole session
    cv_analyzer.close()

Run this file directly to test with your webcam:
    python analyzer.py
"""
import json
import time

import cv2

try:  # works when imported as part of a package ...
    from .confidence import ConfidenceAnalyzer
    from .eye_contact import EyeContactAnalyzer
    from .gestures import GestureAnalyzer
    from .posture import PostureAnalyzer
except ImportError:  # ... and when run directly from this folder
    from confidence import ConfidenceAnalyzer
    from eye_contact import EyeContactAnalyzer
    from gestures import GestureAnalyzer
    from posture import PostureAnalyzer


class CVAnalyzer:
    """Eye contact + posture + gestures + confidence, calibrated together."""

    def __init__(self):
        self.eye_contact = EyeContactAnalyzer()
        self.posture = PostureAnalyzer()
        self.gestures = GestureAnalyzer()
        self.confidence = ConfidenceAnalyzer()
        self._calibrating = False

    # ----- calibration -----------------------------------------------------
    @property
    def calibrating(self):
        return self._calibrating

    @property
    def calibrated(self):
        return (self.eye_contact.calibrated and self.posture.calibrated
                and self.confidence.calibrated)

    def begin_calibration(self):
        """
        Start calibration. The speaker sits upright, looks at the camera with a
        relaxed neutral face, head and both shoulders in view. Keep calling
        analyze_frame() for about 3 seconds, then call finish_calibration().
        """
        self._calibrating = True
        self.posture.begin_calibration()
        self.confidence.begin_calibration()

    def finish_calibration(self, frame_bgr, timestamp_ms=None):
        """
        Finish calibration using the latest frame.
        Returns {"eye_contact": bool, "posture": bool, "confidence": bool, "ok": bool}.
        When it succeeds, the session summary is cleared so results start fresh.
        """
        self._calibrating = False
        eye_ok = self.eye_contact.calibrate(frame_bgr, timestamp_ms)
        posture_ok = self.posture.finish_calibration()
        conf_ok = self.confidence.finish_calibration()
        ok = eye_ok and posture_ok and conf_ok
        if ok:
            self.reset()
        return {"eye_contact": eye_ok, "posture": posture_ok,
                "confidence": conf_ok, "ok": ok}
    def calibrate(self, frame_bgr):
        """
        Quick single-frame calibration (less precise, no confidence baseline).
        Prefer begin_calibration() / finish_calibration().
        """
        eye_ok = self.eye_contact.calibrate(frame_bgr)
        posture_ok = self.posture.calibrate(frame_bgr)
        ok = eye_ok and posture_ok
        if ok:
            self.reset()
        return {"eye_contact": eye_ok, "posture": posture_ok,
                "confidence": None, "ok": ok}

    # ----- analysis --------------------------------------------------------
    def analyze_frame(self, frame_bgr, include_drawing_data=False,
                      timestamp_ms=None):
        """
        Analyse one frame with all analyzers.
        timestamp_ms : time of the frame in milliseconds. Leave it out for a
                       live webcam; pass it when analysing a recorded video.
        Returns {"eye_contact", "posture", "gestures", "confidence"} dicts.
        Landmark and blendshape data (only needed for drawing) are left out
        unless include_drawing_data=True.
        """
        eye = self.eye_contact.analyze_frame(frame_bgr, timestamp_ms)
        face_points = eye.get("landmarks")

        face_box = None
        if face_points:
            xs = [p[0] for p in face_points]
            ys = [p[1] for p in face_points]
            face_box = (min(xs), min(ys), max(xs), max(ys))

        gestures = self.gestures.analyze_frame(
            frame_bgr, timestamp_ms, face_box=face_box)
        hands_active = (gestures["state"] in ("gesturing", "fidgeting")
                        or gestures["touching_face"])
        posture = self.posture.analyze_frame(
            frame_bgr, timestamp_ms, hands_active=hands_active)

        now = time.monotonic() if timestamp_ms is None else timestamp_ms / 1000.0
        confidence = self.confidence.update(
            now, eye.get("blendshapes"), face_points)

        if not include_drawing_data:
            eye.pop("landmarks", None)
            eye.pop("blendshapes", None)
            posture.pop("points", None)
            gestures.pop("landmarks", None)

        return {"eye_contact": eye, "posture": posture,
                "gestures": gestures, "confidence": confidence}
    def summary(self):
        """Combined session report. Safe to call at any time."""
        return {
            "eye_contact": self.eye_contact.summary(),
            "posture": self.posture.summary(),
            "gestures": self.gestures.summary(),
            "confidence": self.confidence.summary(),
        }

    def to_json(self, indent=2):
        return json.dumps(self.summary(), indent=indent)

    def reset(self):
        """Start a new session (keeps the calibration)."""
        self.eye_contact.reset()
        self.posture.reset()
        self.gestures.reset()
        self.confidence.reset()

    def close(self):
        self.eye_contact.close()
        self.posture.close()
        self.gestures.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def analyze_video(path, sample_fps=15.0, calibration_seconds=3.0, max_width=640):
    """
    Analyse a recorded video file and return the combined session report.

    The FIRST `calibration_seconds` of the video are used for calibration: the
    speaker must sit upright, hands down, relaxed neutral face, looking at the
    camera. Everything after that is analysed.

    sample_fps : frames analysed per second of video (higher = more accurate
                 blink counting, but slower)
    max_width  : frames wider than this are shrunk first, which is much faster
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:   # browser recordings often report nonsense
        fps = 30.0
    min_gap_ms = 1000.0 / sample_fps

    calibrated, analyzed = False, 0
    calibration_pending = True
    last_t, last_analyzed = -1.0, -1e9

    with CVAnalyzer() as analyzer:
        analyzer.begin_calibration()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_ms = pos if pos and pos > last_t else last_t + 1000.0 / fps
                last_t = t_ms
                if t_ms - last_analyzed < min_gap_ms * 0.75:
                    continue
                last_analyzed = t_ms

                h, w = frame.shape[:2]
                if w > max_width:
                    frame = cv2.resize(frame, (max_width, int(h * max_width / w)))

                if calibration_pending and t_ms >= calibration_seconds * 1000:
                    calibrated = analyzer.finish_calibration(frame, int(t_ms))["ok"]
                    calibration_pending = False
                    continue

                analyzer.analyze_frame(frame, timestamp_ms=int(t_ms))
                analyzed += 1
        finally:
            cap.release()

        report = analyzer.summary()

    report["meta"] = {
        "calibrated": calibrated,
        "video_seconds": round(max(last_t, 0) / 1000, 1),
        "frames_analyzed": analyzed,
        "calibration_seconds": calibration_seconds,
    }
    return report

def run_demo():
    """Webcam test with an on-screen display (not used by the app)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    green, red, orange = (0, 200, 0), (0, 0, 255), (0, 165, 255)
    blue, yellow = (255, 200, 0), (0, 255, 255)
    calib_end, message, message_until = 0.0, "", 0.0
    fps, last_time = 0.0, time.monotonic()

    with CVAnalyzer() as cv_analyzer:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                now = time.monotonic()

                result = cv_analyzer.analyze_frame(frame, include_drawing_data=True)
                eye, posture = result["eye_contact"], result["posture"]
                gestures, conf = result["gestures"], result["confidence"]

                if cv_analyzer.calibrating and now >= calib_end:
                    r = cv_analyzer.finish_calibration(frame)
                    if r["ok"]:
                        message = "Calibrated!"
                    else:
                        failed = [k for k in ("eye_contact", "posture", "confidence")
                                  if not r[k]]
                        message = ("Calibration failed (" + ", ".join(failed) +
                                   "): show face and both shoulders")
                    message_until = now + 3

                s = cv_analyzer.summary()

                # Draw face, hands and posture points
                for x, y in (eye.get("landmarks") or []):
                    cv2.circle(frame, (int(x * w), int(y * h)), 1, (255, 255, 0), -1)
                for hand in gestures.get("landmarks", []):
                    for x, y in hand:
                        cv2.circle(frame, (int(x * w), int(y * h)), 3, green, -1)
                pts = posture.get("points")
                if pts:
                    px = {k: (int(x * w), int(y * h)) for k, (x, y) in pts.items()}
                    p_color = green if posture["state"] == "good" else red
                    cv2.line(frame, px["l_shoulder"], px["r_shoulder"], p_color, 2)
                    cv2.line(frame, px["l_ear"], px["r_ear"], p_color, 2)

                # Text panel
                eye_color = (green if eye["state"] == "looking"
                             else red if eye["state"] == "away" else orange)
                posture_color = (green if posture["state"] == "good"
                                 else red if posture["state"] == "poor" else orange)
                gesture_color = {"gesturing": green, "fidgeting": red,
                                 "still": blue, "none": orange}[gestures["state"]]
                conf_color = {"confident": green, "neutral": blue,
                              "nervous": red, "calibrating": yellow,
                              "no_face": orange}[conf["state"]]
                score = posture.get("score")

                lines = [
                    (f"Eye: {eye['label']}  ({s['eye_contact']['eye_contact_pct']:.0f}%)",
                     eye_color),
                    (f"Posture: {posture['label']}  (score {score if score is not None else '-'})",
                     posture_color),
                    (f"Hands: {gestures['label']}", gesture_color),
                    (f"Face: {conf['label']}" +
                     (f"  ({conf['confidence']})" if conf["confidence"] is not None else ""),
                     conf_color),
                ]
                for i, (text, color) in enumerate(lines):
                    cv2.putText(frame, text, (20, 32 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if cv_analyzer.calibrating:
                    left = max(0.0, calib_end - now)
                    cv2.putText(frame, f"Calibrating... hold still {left:.1f}s",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, yellow, 2)
                elif not cv_analyzer.calibrated:
                    cv2.putText(frame,
                                "Sit upright, relaxed face, look at camera, press C",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
                if now < message_until:
                    cv2.putText(frame, message, (20, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, orange, 2)

                fps = 0.9 * fps + 0.1 * (1 / max(now - last_time, 1e-6))
                last_time = now
                cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.imshow("CV Analyzer", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c') and not cv_analyzer.calibrating:
                    cv_analyzer.begin_calibration()
                    calib_end = time.monotonic() + 3.0
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(cv_analyzer.to_json())


if __name__ == "__main__":
    run_demo()