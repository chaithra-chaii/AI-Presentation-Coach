import json
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"
)

STATES = ("looking", "away", "no_face")
LABELS = {
    "looking": "Looking at camera",
    "away": "Looking away",
    "no_face": "No face",
}

# If two frames are further apart than this, the gap counts as this long
MAX_GAP_SECONDS = 0.5


def _eye_ratio(lm, iris, outer, inner):
    """Where the iris sits between the two eye corners (0.5 = centre)."""
    width = lm[inner].x - lm[outer].x
    if width == 0:
        return 0.5
    return (lm[iris].x - lm[outer].x) / width


def _head_pitch(lm):
    """Nose position between forehead and chin (changes when the head tilts)."""
    height = lm[152].y - lm[10].y
    if height == 0:
        return 0.5
    return (lm[1].y - lm[10].y) / height


def _eye_vertical_score(blendshapes):
    """0 = eyes straight, 1 = eyes fully up or down."""
    s = {b.category_name: b.score for b in blendshapes}
    down = (s.get("eyeLookDownLeft", 0) + s.get("eyeLookDownRight", 0)) / 2
    up = (s.get("eyeLookUpLeft", 0) + s.get("eyeLookUpRight", 0)) / 2
    return max(down, up)


class EyeContactAnalyzer:
    """Tracks whether the speaker is looking at the camera across a session."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, head_tolerance=0.12,
                 gaze_tolerance=0.15, pitch_tolerance=0.06,
                 eye_vertical_limit=0.35):
        """
        head_tolerance     : head turned left/right
        gaze_tolerance     : eyes looking left/right
        pitch_tolerance    : head tilted up/down (needs calibrate())
        eye_vertical_limit : eyes looking up/down
        Higher numbers = more forgiving, lower = stricter.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Face model not found at: {model_path}\n"
                "Download face_landmarker.task into the models folder."
            )
        self.head_tolerance = head_tolerance
        self.gaze_tolerance = gaze_tolerance
        self.pitch_tolerance = pitch_tolerance
        self.eye_vertical_limit = eye_vertical_limit

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        # MediaPipe needs timestamps that always increase, so these are
        # never reset, even when a new session starts.
        self._origin = time.monotonic()
        self._last_ts = -1
        self.baseline_pitch = None
        self.reset()

    def reset(self):
        """Start a new session (clears the summary, keeps the calibration)."""
        self._durations = {s: 0.0 for s in STATES}
        self._prev_ts = None

    @property
    def calibrated(self):
        return self.baseline_pitch is not None

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
        Call while the speaker looks straight at the camera.
        Returns True if a face was found and the calibration was saved.
        """
        result, _ = self._detect(frame_bgr)
        if not result.face_landmarks:
            return False
        self.baseline_pitch = _head_pitch(result.face_landmarks[0])
        return True

    def _check(self, lm, blendshapes):
        """Returns (looking, eye_vertical_score, pitch_difference)."""
        face_width = lm[454].x - lm[234].x
        if face_width == 0:
            return False, 0.0, 0.0
        head = (lm[1].x - lm[234].x) / face_width

        right = _eye_ratio(lm, 468, 33, 133)
        left = _eye_ratio(lm, 473, 362, 263)
        gaze = (right + left) / 2

        v_score = _eye_vertical_score(blendshapes) if blendshapes else 0.0

        if self.baseline_pitch is None:
            pitch_diff = 0.0
        else:
            pitch_diff = abs(_head_pitch(lm) - self.baseline_pitch)

        looking = (abs(head - 0.5) < self.head_tolerance
                   and abs(gaze - 0.5) < self.gaze_tolerance
                   and v_score < self.eye_vertical_limit
                   and pitch_diff < self.pitch_tolerance)
        return looking, v_score, pitch_diff

    def analyze_frame(self, frame_bgr, timestamp_ms=None):
        """
        Analyse one frame.
        Returns a dict: state, label, looking, calibrated,
        eye_vertical_score, head_tilt_diff
        """
        result, ts = self._detect(frame_bgr, timestamp_ms)

        dt = 0.0
        if self._prev_ts is not None:
            dt = min((ts - self._prev_ts) / 1000.0, MAX_GAP_SECONDS)
        self._prev_ts = ts

        v_score, pitch_diff = 0.0, 0.0
        points = None
        shapes = None
        if not result.face_landmarks:
            state = "no_face"
        else:
            blend = result.face_blendshapes[0] if result.face_blendshapes else None
            looking, v_score, pitch_diff = self._check(
                result.face_landmarks[0], blend)
            state = "looking" if looking else "away"
            points = [(p.x, p.y) for p in result.face_landmarks[0]]
            shapes = ({b.category_name: b.score for b in blend}
                      if blend else {})

        self._durations[state] += dt

        return {
            "state": state,
            "label": LABELS[state],
            "looking": state == "looking",
            "calibrated": self.calibrated,
            "eye_vertical_score": round(v_score, 3),
            "head_tilt_diff": round(pitch_diff, 3),
            "landmarks": points,
            "blendshapes": shapes,
        }

    def summary(self):
        """Session results. Eye contact % counts only the time a face was seen."""
        total = sum(self._durations.values())
        face_time = self._durations["looking"] + self._durations["away"]
        if total <= 0:
            return {"total_seconds": 0, "eye_contact_pct": 0,
                    "face_detected_pct": 0}
        return {
            "total_seconds": round(total, 1),
            "eye_contact_pct": round(
                self._durations["looking"] / face_time * 100, 1) if face_time else 0,
            "face_detected_pct": round(face_time / total * 100, 1),
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

    colors = {"looking": (0, 200, 0), "away": (0, 0, 255),
              "no_face": (0, 165, 255)}

    with EyeContactAnalyzer() as analyzer:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = analyzer.analyze_frame(frame)
                s = analyzer.summary()

                cv2.putText(frame, result["label"], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            colors[result["state"]], 2)
                cv2.putText(frame, f"Eye contact: {s['eye_contact_pct']:.0f}%",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.putText(frame,
                            f"eyes up/down: {result['eye_vertical_score']:.2f}  "
                            f"head tilt: {result['head_tilt_diff']:.2f}",
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)
                if not analyzer.calibrated:
                    cv2.putText(frame, "Look at camera and press C to calibrate",
                                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)
                cv2.imshow("Eye Contact", frame)

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