import os
import time
import math
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "hand_landmarker.task"
)

WINDOW_SECONDS = 1.5
EMA_ALPHA = 0.35

STILL_SPEED = 0.06
GESTURE_SPEED = 0.12
REPETITIVE_SPEED = 0.30

MIN_MOVEMENT = 0.006
MIN_HISTORY_POINTS = 6

STATES = {
    "none": ("Hands not visible", (0, 165, 255)),
    "still": ("Hands relatively still", (255, 200, 0)),
    "gesturing": ("Active hand movement", (0, 180, 0)),
    "fidgeting": ("Repetitive movement detected", (0, 0, 255)),
}


def analyze_history(history):
    """
    Returns:
        speed: normalized wrist movement per second
        reversals: significant direction changes
        efficiency: net distance / total path distance
    """

    if len(history) < MIN_HISTORY_POINTS:
        return 0.0, 0, 1.0

    points = list(history)
    duration = points[-1][0] - points[0][0]

    if duration <= 0:
        return 0.0, 0, 1.0

    path_length = 0.0
    directions = []

    for i in range(1, len(points)):
        dx = points[i][1] - points[i - 1][1]
        dy = points[i][2] - points[i - 1][2]

        distance = math.hypot(dx, dy)
        path_length += distance

        if distance >= MIN_MOVEMENT:
            directions.append((dx / distance, dy / distance))

    speed = path_length / duration

    net_dx = points[-1][1] - points[0][1]
    net_dy = points[-1][2] - points[0][2]
    net_distance = math.hypot(net_dx, net_dy)

    efficiency = net_distance / max(path_length, 1e-6)

    reversals = 0

    for i in range(1, len(directions)):
        previous = directions[i - 1]
        current = directions[i]

        dot = (
            previous[0] * current[0]
            + previous[1] * current[1]
        )

        if dot < -0.25:
            reversals += 1

    return speed, reversals, efficiency


def classify_hand(speed, reversals, efficiency):
    """
    Classifies observed movement patterns.
    It does not diagnose nervousness or intent.
    """

    if speed < STILL_SPEED:
        return "still"

    repetitive = (
        speed >= REPETITIVE_SPEED
        and reversals >= 3
        and efficiency < 0.45
    )

    if repetitive:
        return "fidgeting"

    if speed >= GESTURE_SPEED:
        return "gesturing"

    return "still"


def main():

    if not os.path.isfile(MODEL_PATH):
        print("ERROR: Hand landmark model not found.")
        print("Expected location:")
        print(MODEL_PATH)
        return

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.HandLandmarker.create_from_options(
        options
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        landmarker.close()
        return

    
    histories = {}
    smooth_positions = {}
    previous_states = {}

    counts = {name: 0 for name in STATES}
    total_frames = 0
    start = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam frame.")
                break

            height, width = frame.shape[:2]
            now = time.monotonic() - start

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = landmarker.detect_for_video(
                image, int(now * 1000)
            )

            visible_states = []
            active_ids = set()

            for index, hand in enumerate(result.hand_landmarks):

                
                hand_id = str(index)

                if (
                    result.handedness
                    and index < len(result.handedness)
                    and result.handedness[index]
                ):
                    hand_id = (
                        result.handedness[index][0].category_name
                    )

                active_ids.add(hand_id)

                
                wrist = hand[0]
                raw_x = wrist.x
                raw_y = wrist.y

            
                if hand_id not in smooth_positions:
                    smooth_positions[hand_id] = (raw_x, raw_y)
                else:
                    old_x, old_y = smooth_positions[hand_id]

                    raw_x = (
                        EMA_ALPHA * raw_x
                        + (1 - EMA_ALPHA) * old_x
                    )
                    raw_y = (
                        EMA_ALPHA * raw_y
                        + (1 - EMA_ALPHA) * old_y
                    )

                    smooth_positions[hand_id] = (raw_x, raw_y)

                if hand_id not in histories:
                    histories[hand_id] = deque()

                history = histories[hand_id]
                history.append((now, raw_x, raw_y))

                while (
                    history
                    and now - history[0][0] > WINDOW_SECONDS
                ):
                    history.popleft()

                speed, reversals, efficiency = analyze_history(
                    history
                )

                if len(history) < MIN_HISTORY_POINTS:
                    state = "still"
                else:
                    state = classify_hand(
                        speed, reversals, efficiency
                    )

                visible_states.append(state)
                previous_states[hand_id] = state

                # Draw landmarks and wrist.
                for point in hand:
                    px = int(point.x * width)
                    py = int(point.y * height)

                    cv2.circle(
                        frame, (px, py), 2, (0, 220, 0), -1
                    )

                wx = int(raw_x * width)
                wy = int(raw_y * height)

                cv2.circle(
                    frame, (wx, wy), 8, (255, 100, 0), -1
                )

                cv2.putText(
                    frame,
                    f"{hand_id}: {state}",
                    (max(10, wx - 50), max(25, wy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    STATES[state][1],
                    2,
                )

            
            for hand_id in list(histories):
                if hand_id not in active_ids:
                    histories.pop(hand_id, None)
                    smooth_positions.pop(hand_id, None)
                    previous_states.pop(hand_id, None)

            
            if not visible_states:
                overall_state = "none"
            elif "fidgeting" in visible_states:
                overall_state = "fidgeting"
            elif "gesturing" in visible_states:
                overall_state = "gesturing"
            else:
                overall_state = "still"

            total_frames += 1
            counts[overall_state] += 1

            visible_pct = (
                sum(counts[s] for s in
                    ("still", "gesturing", "fidgeting"))
                / total_frames * 100
            )

            gesture_pct = (
                counts["gesturing"] / total_frames * 100
            )

            repetitive_pct = (
                counts["fidgeting"] / total_frames * 100
            )

            label, color = STATES[overall_state]

           
            cv2.rectangle(
                frame, (10, 10), (510, 165), (35, 35, 35), -1
            )

            cv2.putText(
                frame, label, (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
            )

            cv2.putText(
                frame, f"Hands detected: {len(visible_states)}",
                (25, 78), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

            cv2.putText(
                frame, f"Hands visible: {visible_pct:.0f}%",
                (25, 108), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

            cv2.putText(
                frame, f"Active movement: {gesture_pct:.0f}%",
                (25, 136), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1
            )

            cv2.putText(
                frame, f"Repetitive movement: {repetitive_pct:.0f}%",
                (25, 160), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            cv2.imshow("AI Presentation Coach - Gesture Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    if total_frames:
        print("\n--- Gesture Session Summary ---")
        print(f"Hands visible: {visible_pct:.1f}%")
        print(f"Active movement: {gesture_pct:.1f}%")
        print(f"Repetitive movement: {repetitive_pct:.1f}%")
        print("Note: These are movement estimates, not emotion scores.")


if __name__ == "__main__":
    main()

