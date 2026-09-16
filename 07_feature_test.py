import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import joblib
import json
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gaze_features import get_features, FEATURE_VERSION


# ==========================================================
# 1. LOAD MODEL & METADATA
# ==========================================================

with open("gaze_model_metadata.json", "r") as f:
    metadata = json.load(f)

# Verify feature compatibility
if metadata.get("feature_version") != FEATURE_VERSION:
    print(f"ERROR: Model version mismatch ({metadata.get('feature_version')} vs {FEATURE_VERSION}).")
    exit()

model = joblib.load("gaze_model.pkl")
USE_HEAD_POSE = metadata.get("use_head_pose", False)

print(f"Loaded Model: {metadata['model_type']} (CV Mean Error: {metadata['cv_mean_error_px']:.1f}px)")


# ==========================================================
# 2. SCREEN DIMENSIONS
# ==========================================================

root = tk.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()


# ==========================================================
# 3. MEDIAPIPE INITIALIZATION
# ==========================================================

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_facial_transformation_matrixes=USE_HEAD_POSE,
)

detector = vision.FaceLandmarker.create_from_options(options)


def extract_features(result):
    if not result.face_landmarks:
        return None
    matrix = None
    if USE_HEAD_POSE and result.facial_transformation_matrixes:
        matrix = result.facial_transformation_matrixes[0]
    return get_features(result.face_landmarks[0], matrix)


# ==========================================================
# 4. CAMERA & DISPLAY SETUP
# ==========================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    detector.close()
    exit("ERROR: Could not open webcam.")

WINDOW_NAME = "Real-Time Gaze Tracking Application"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Smoothing parameters (EMA)
smooth_x, smooth_y = None, None
alpha = 0.20  # Lower = smoother/slower, Higher = faster/more jittery

timestamp_ms = 0

print("\n==========================================")
print("REAL-TIME GAZE APPLICATION ACTIVE")
print("==========================================")
print("Look around your screen to track your gaze point.")
print("Press 'Q' to quit.\n")


# ==========================================================
# 5. MAIN TRACKING LOOP
# ==========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    # Canvas to draw fullscreen gaze output
    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    features = extract_features(result)
    if features is not None:
        feat_array = np.array(features).reshape(1, -1)
        pred = model.predict(feat_array)[0]

        raw_x, raw_y = pred[0], pred[1]

        # Apply Exponential Moving Average (EMA) Filter
        if smooth_x is None or smooth_y is None:
            smooth_x, smooth_y = raw_x, raw_y
        else:
            smooth_x = alpha * raw_x + (1 - alpha) * smooth_x
            smooth_y = alpha * raw_y + (1 - alpha) * smooth_y

        gaze_x = int(np.clip(smooth_x, 0, SCREEN_WIDTH - 1))
        gaze_y = int(np.clip(smooth_y, 0, SCREEN_HEIGHT - 1))

        # Render Gaze Cursor
        cv2.circle(canvas, (gaze_x, gaze_y), 25, (0, 255, 0), -1)
        cv2.circle(canvas, (gaze_x, gaze_y), 8, (255, 255, 255), -1)

        # Display Coordinates
        cv2.putText(
            canvas,
            f"Gaze Coordinates: ({gaze_x}, {gaze_y})",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================================
# 6. CLEANUP
# ==========================================================

cap.release()
detector.close()
cv2.destroyAllWindows()