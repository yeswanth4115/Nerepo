import json
import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gaze_features import FEATURE_VERSION, get_features

# Load model and metadata
with open("gaze_model_metadata.json", "r") as f:
    metadata = json.load(f)

if metadata.get("feature_version") != FEATURE_VERSION:
    print(
        "ERROR: Model feature version mismatch. Model expects"
        f" '{metadata.get('feature_version')}', but script has"
        f" '{FEATURE_VERSION}'."
    )
    exit()

model = joblib.load("gaze_model.pkl")
USE_HEAD_POSE = metadata.get("use_head_pose", False)

# Screen setup
root = tk.Tk()
root.withdraw()
SCREEN_WIDTH, SCREEN_HEIGHT = (
    root.winfo_screenwidth(),
    root.winfo_screenheight(),
)
root.destroy()

# MediaPipe setup
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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    detector.close()
    exit("ERROR: Could not open webcam.")

WINDOW_NAME = "Gaze Estimation"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)

smooth_x, smooth_y = None, None
alpha = 0.25  # Exponential Moving Average factor
timestamp_ms = 0

print(
    f"Loaded {metadata['model_type']} model (CV Mean Error:"
    f" {metadata['cv_mean_error_px']:.1f}px). Press 'Q' to quit."
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    if result.face_landmarks:
        matrix = (
            result.facial_transformation_matrixes[0]
            if (USE_HEAD_POSE and result.facial_transformation_matrixes)
            else None
        )
        features = get_features(result.face_landmarks[0], matrix)

        if features is not None:
            pred = model.predict(np.array(features).reshape(1, -1))[0]
            raw_x, raw_y = pred[0], pred[1]

            if smooth_x is None or smooth_y is None:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = alpha * raw_x + (1 - alpha) * smooth_x
                smooth_y = alpha * raw_y + (1 - alpha) * smooth_y

            gaze_x = int(np.clip(smooth_x, 0, SCREEN_WIDTH - 1))
            gaze_y = int(np.clip(smooth_y, 0, SCREEN_HEIGHT - 1))

            cv2.circle(canvas, (gaze_x, gaze_y), 20, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f"Gaze: ({gaze_x}, {gaze_y})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    cv2.imshow(WINDOW_NAME, canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()