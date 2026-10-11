import json
import time

import cv2

try:  # works when imported as part of a package ...
    from .eye_contact import EyeContactAnalyzer
    from .gestures import GestureAnalyzer
    from .posture import PostureAnalyzer
except ImportError:  # ... and when run directly from this folder
    from eye_contact import EyeContactAnalyzer
    from gestures import GestureAnalyzer
    from posture import PostureAnalyzer


class CVAnalyzer:
    
    def __init__(self):
        self.eye_contact = EyeContactAnalyzer()
        self.posture = PostureAnalyzer()
        self.gestures = GestureAnalyzer()

    def calibrate(self, frame_bgr):
       
        eye_ok = self.eye_contact.calibrate(frame_bgr)
        posture_ok = self.posture.calibrate(frame_bgr)
        ok = eye_ok and posture_ok
        if ok:
            self.reset()
        return {"eye_contact": eye_ok, "posture": posture_ok, "ok": ok}

    @property
    def calibrated(self):
        return self.eye_contact.calibrated and self.posture.calibrated

    def analyze_frame(self, frame_bgr, include_drawing_data=False):
       
        eye = self.eye_contact.analyze_frame(frame_bgr)
        posture = self.posture.analyze_frame(frame_bgr)
        face_box = None
        face_pts = eye.get("landmarks")
        if face_pts:
            xs = [p[0] for p in face_pts]
            ys = [p[1] for p in face_pts]
            face_box = (min(xs), min(ys), max(xs), max(ys))
        gestures = self.gestures.analyze_frame(frame_bgr, face_box=face_box)

        if not include_drawing_data:
            eye.pop("landmarks", None)
            posture.pop("points", None)
            gestures.pop("landmarks", None)

        return {"eye_contact": eye, "posture": posture, "gestures": gestures}

    def summary(self):
        
        return {
            "eye_contact": self.eye_contact.summary(),
            "posture": self.posture.summary(),
            "gestures": self.gestures.summary(),
        }

    def to_json(self, indent=2):
        return json.dumps(self.summary(), indent=indent)

    def reset(self):
        
        self.eye_contact.reset()
        self.posture.reset()
        self.gestures.reset()

    def close(self):
        self.eye_contact.close()
        self.posture.close()
        self.gestures.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_demo():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    green, red, orange = (0, 200, 0), (0, 0, 255), (0, 165, 255)
    message, message_until = "", 0.0
    fps, last_time = 0.0, time.monotonic()

    with CVAnalyzer() as cv_analyzer:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]

                result = cv_analyzer.analyze_frame(frame, include_drawing_data=True)
                eye, posture, gestures = (result["eye_contact"],
                                          result["posture"], result["gestures"])
                s = cv_analyzer.summary()

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

                
                eye_color = (green if eye["state"] == "looking"
                             else red if eye["state"] == "away" else orange)
                posture_color = (green if posture["state"] == "good"
                                 else red if posture["state"] == "poor" else orange)
                gesture_color = {"gesturing": green, "fidgeting": red,
                                 "still": (255, 200, 0), "none": orange}[gestures["state"]]

                lines = [
                    (f"Eye: {eye['label']}  ({s['eye_contact']['eye_contact_pct']:.0f}%)", eye_color),
                    (f"Posture: {posture['label']}  ({s['posture']['good_posture_pct']:.0f}%)", posture_color),
                    (f"Hands: {gestures['label']}  (gesturing {s['gestures']['gesturing_pct']:.0f}%)", gesture_color),
                ]
                for i, (text, color) in enumerate(lines):
                    cv2.putText(frame, text, (20, 35 + i * 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if not cv_analyzer.calibrated:
                    cv2.putText(frame, "Sit upright, look at camera, press C",
                                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 255, 255), 2)
                if time.monotonic() < message_until:
                    cv2.putText(frame, message, (20, 175),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, orange, 2)

                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1 / max(now - last_time, 1e-6))
                last_time = now
                cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                cv2.imshow("CV Analyzer", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('c'):
                    r = cv_analyzer.calibrate(frame)
                    if not r["ok"]:
                        message = "Calibration failed: face and both shoulders must be visible"
                        message_until = time.monotonic() + 3
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(cv_analyzer.to_json())


if __name__ == "__main__":
    run_demo()