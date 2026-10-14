# Computer Vision Module

Analyses a speaker's **eye contact**, **posture** and **hand gestures** from webcam frames.
Part of the AI Presentation Coach project (CV module by Mem 2).

## Files

| File | Purpose |
|---|---|
| `analyzer.py` | **Start here.** Runs all three analyses together and returns one report |
| `eye_contact.py` | Is the speaker looking at the camera? |
| `posture.py` | Shoulders, head tilt, slouching, leaning forward |
| `gestures.py` | Are hands visible, gesturing, still or fidgeting? |
| `models/` | MediaPipe model files (needed, do not delete) |

## Setup

```bash
pip install -r computer_vision/requirements.txt
```

Tested with Python 3.14 and mediapipe 1.0.1.

## Quick test (webcam window)

```bash
python computer_vision/analyzer.py
```

Sit upright, look at the camera and press `C` to calibrate. Press `Q` to quit. The final report is printed as JSON.

## Using it in the app

## Using it in the app

```python
from analyzer import CVAnalyzer

cv = CVAnalyzer()

# 1. Calibrate (about 3 seconds): speaker sits upright, relaxed neutral face,
#    looks at the camera, head and both shoulders visible
cv.begin_calibration()
# ... keep calling cv.analyze_frame(frame) for ~3 seconds ...
status = cv.finish_calibration(frame)   # {"eye_contact": True, "posture": True, "confidence": True, "ok": True}

# 2. For every frame during the session
live = cv.analyze_frame(frame)          # per-frame result, good for live hints

# 3. At the end of the session
report = cv.summary()                   # one dict for the whole session
print(cv.to_json())

cv.close()
```

- `frame` is a **BGR image as a numpy array** (the format OpenCV gives, e.g. from `cv2.VideoCapture`).
- `finish_calibration()` returns `ok: False` if a face or both shoulders were not visible. Ask the speaker to try again.
- `cv.reset()` starts a new session and keeps the calibration.
- `cv.calibrate(frame)` is a quick single-frame calibration (less precise, no confidence baseline).
- Without calibration the analyzers still run, but with fewer checks (no slouching, leaning or sway detection, and the confidence estimate has no personal baseline).

## Report format (`summary()`)

```json
{
  "eye_contact":  {"total_seconds": 31.1, "eye_contact_pct": 63.1, "face_detected_pct": 100.0},
  "posture":      {"total_seconds": 31.1, "good_posture_pct": 53.7, "avg_posture_score": 69.3,
                   "visible_pct": 100.0, "shoulders_uneven_pct": 3.5, "head_tilted_pct": 10.6,
                   "slouching_pct": 34.0, "leaning_forward_pct": 21.1, "swaying_pct": 10.0},
  "gestures":     {"total_seconds": 31.1, "hands_visible_pct": 79.4, "gesturing_pct": 44.9,
                   "still_pct": 34.2, "fidgeting_pct": 0.3, "face_touch_pct": 2.1},
  "confidence":   {"total_seconds": 31.1, "avg_confidence": 72.5, "confident_pct": 60.0,
                   "neutral_pct": 30.0, "nervous_pct": 10.0,
                   "top_cues": ["frequent blinking", "gaze darting around"]}
}
```

All percentages are **shares of time**, not frames.

- `eye_contact_pct`: share of time looking at the camera, counted only while a face was detected
- `face_detected_pct` / `visible_pct`: share of the session the speaker was in view (low = camera blocked or speaker out of frame)
- `good_posture_pct`: share of visible time with no posture problem. `avg_posture_score` is 0-100. The five problem percentages can overlap
- `gesturing_pct`: natural movement. `still_pct`: hands visible but not moving. `fidgeting_pct`: fast, repetitive movement. `face_touch_pct`: a hand near the face
- `avg_confidence`: 0-100 **estimate based on visible cues** (blinking, gaze darting, head movement, pressed lips, tense brows, wide eyes, smiles). It is **not** a measure of the speaker's real emotions. `top_cues` lists the most common cues in the session

## Notes and limits

- Built for **live webcam frames**. By default it times frames using the computer's clock. For a recorded video, pass the video's own time with `analyze_frame(frame, timestamp_ms=...)` on the individual analyzers.
- The detection limits (angles, speeds) are tuned for a normal desk webcam and should be tested with other people. They can be changed in each analyzer's constructor.
- First run is slower because three models load at once.