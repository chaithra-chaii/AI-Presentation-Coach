import math
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

# Tuning values
SHOULDER_TILT_LIMIT = 8     # degrees: shoulders uneven
HEAD_TILT_LIMIT = 10        # degrees: head leaning sideways
SLOUCH_RATIO = 0.90         # head sinks: below 90% of upright height
LEAN_RATIO = 1.12           # head leans toward camera: 12% bigger than upright


def angle_deg(p1, p2, w, h):
    """Tilt of the line between two points. 0 = perfectly level."""
    dx = (p2.x - p1.x) * w
    dy = (p2.y - p1.y) * h
    ang = math.degrees(math.atan2(dy, dx))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return ang


def posture_metrics(lm, w, h):
    l_sh, r_sh = lm[11], lm[12]
    l_ear, r_ear = lm[7], lm[8]

    shoulder_tilt = abs(angle_deg(l_sh, r_sh, w, h))
    head_tilt = abs(angle_deg(l_ear, r_ear, w, h))

    # Ear-to-shoulder height, relative to shoulder width
    sh_width = math.hypot((l_sh.x - r_sh.x) * w, (l_sh.y - r_sh.y) * h)
    sh_mid_y = (l_sh.y + r_sh.y) / 2 * h
    ear_mid_y = (l_ear.y + r_ear.y) / 2 * h
    height_ratio = (sh_mid_y - ear_mid_y) / sh_width if sh_width else 0

    # Head size on screen (ear to ear, in pixels): grows when you lean in
    ear_width = math.hypot((l_ear.x - r_ear.x) * w, (l_ear.y - r_ear.y) * h)

    return shoulder_tilt, head_tilt, height_ratio, ear_width


def shoulders_visible(lm):
    return min((lm[11].visibility or 0), (lm[12].visibility or 0)) > 0.5


def draw_points(frame, lm, w, h, color):
    pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in (7, 8, 11, 12)}
    cv2.line(frame, pts[11], pts[12], color, 2)   # shoulders
    cv2.line(frame, pts[7], pts[8], color, 2)     # ears
    for p in pts.values():
        cv2.circle(frame, p, 6, color, -1)


def main():
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start = time.time()
    total_frames = 0
    good_frames = 0
    baseline_ratio = None
    baseline_ear = None
    pct = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        text, color = "No person", (0, 165, 255)
        info = ""
        ratio, ear_w = None, None

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            if not shoulders_visible(lm):
                text, color = "Show both shoulders", (0, 165, 255)
            else:
                shoulder_tilt, head_tilt, ratio, ear_w = posture_metrics(lm, w, h)
                problems = []
                if shoulder_tilt > SHOULDER_TILT_LIMIT:
                    problems.append("Shoulders uneven")
                if head_tilt > HEAD_TILT_LIMIT:
                    problems.append("Head tilted")
                if baseline_ratio and ratio < baseline_ratio * SLOUCH_RATIO:
                    problems.append("Slouching")
                if baseline_ear and ear_w > baseline_ear * LEAN_RATIO:
                    problems.append("Leaning forward")

                total_frames += 1
                if problems:
                    text, color = ", ".join(problems), (0, 0, 255)
                else:
                    good_frames += 1
                    text, color = "Good posture", (0, 200, 0)

                info = (f"sh tilt: {shoulder_tilt:.0f}  "
                        f"head tilt: {head_tilt:.0f}  "
                        f"height: {ratio:.2f}  head size: {ear_w:.0f}")
                draw_points(frame, lm, w, h, color)

        pct = (good_frames / total_frames * 100) if total_frames else 0

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Good posture: {pct:.0f}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, info, (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if baseline_ratio is None:
            cv2.putText(frame, "Sit upright and press C to calibrate",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"upright: height {baseline_ratio:.2f}, head size {baseline_ear:.0f}",
                        (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
        cv2.imshow("Posture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and ratio is not None:
            baseline_ratio = ratio
            baseline_ear = ear_w
            total_frames = 0
            good_frames = 0

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final good posture: {pct:.0f}%")


if __name__ == "__main__":
    main()