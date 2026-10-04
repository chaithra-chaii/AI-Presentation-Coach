import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")

# Tuning values: raise = more forgiving, lower = stricter
HEAD_TOLERANCE = 0.12       # head turned left/right
GAZE_TOLERANCE = 0.15       # eyes looking left/right
PITCH_TOLERANCE = 0.06      # head tilted up/down
EYE_VERTICAL_LIMIT = 0.35   # eyes looking up/down


def eye_ratio(lm, iris, outer, inner):
    """Where the iris sits between the two eye corners (0.5 = centre)."""
    width = lm[inner].x - lm[outer].x
    if width == 0:
        return 0.5
    return (lm[iris].x - lm[outer].x) / width


def head_pitch(lm):
    """Nose position between forehead and chin (changes when head tilts)."""
    height = lm[152].y - lm[10].y
    if height == 0:
        return 0.5
    return (lm[1].y - lm[10].y) / height


def eye_vertical_score(blendshapes):
    """0 = eyes straight, 1 = eyes fully up or down."""
    s = {b.category_name: b.score for b in blendshapes}
    down = (s.get("eyeLookDownLeft", 0) + s.get("eyeLookDownRight", 0)) / 2
    up = (s.get("eyeLookUpLeft", 0) + s.get("eyeLookUpRight", 0)) / 2
    return max(down, up)


def check_eye_contact(lm, blendshapes, baseline_pitch):
    """Returns (looking_at_camera, eye_vertical_score, pitch_difference)."""
    # Head turn left/right
    face_width = lm[454].x - lm[234].x
    if face_width == 0:
        return False, 0, 0
    head = (lm[1].x - lm[234].x) / face_width

    # Eyes left/right
    right = eye_ratio(lm, 468, 33, 133)
    left = eye_ratio(lm, 473, 362, 263)
    gaze = (right + left) / 2

    # Eyes up/down
    v_score = eye_vertical_score(blendshapes)

    # Head tilt up/down (only after pressing 'c')
    if baseline_pitch is None:
        pitch_diff = 0
    else:
        pitch_diff = abs(head_pitch(lm) - baseline_pitch)

    ok = (abs(head - 0.5) < HEAD_TOLERANCE
          and abs(gaze - 0.5) < GAZE_TOLERANCE
          and v_score < EYE_VERTICAL_LIMIT
          and pitch_diff < PITCH_TOLERANCE)
    return ok, v_score, pitch_diff


def main():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start = time.time()
    total_frames = 0
    contact_frames = 0
    baseline_pitch = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        v_score, pitch_diff = 0, 0
        if result.face_landmarks:
            total_frames += 1
            lm = result.face_landmarks[0]
            looking, v_score, pitch_diff = check_eye_contact(
                lm, result.face_blendshapes[0], baseline_pitch)
            if looking:
                contact_frames += 1
            text = "Looking at camera" if looking else "Looking away"
            color = (0, 200, 0) if looking else (0, 0, 255)
        else:
            text = "No face"
            color = (0, 165, 255)

        pct = (contact_frames / total_frames * 100) if total_frames else 0

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Eye contact: {pct:.0f}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"eyes up/down: {v_score:.2f}  head tilt: {pitch_diff:.2f}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if baseline_pitch is None:
            cv2.putText(frame, "Look at camera and press C to calibrate",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Eye Contact", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and result.face_landmarks:
            baseline_pitch = head_pitch(result.face_landmarks[0])
            total_frames = 0
            contact_frames = 0

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final eye contact: {pct:.0f}%")


if __name__ == "__main__":
    main()