import json
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_features import (
    FEATURE_VERSION,
    estimate_gaze_confidence,
    extract_eye_metrics,
    get_features,
)
from gaze_kalman import KalmanGazeFilter


MODEL_FILE = "gaze_model.pkl"
METADATA_FILE = "gaze_model_metadata.json"
MODEL_PATH = "models/face_landmarker.task"

WINDOW_NAME = "Gaze Estimation"

KALMAN_PROCESS_NOISE = 800.0
KALMAN_MEASUREMENT_NOISE = 225.0
KALMAN_DEADBAND = 1.0
KALMAN_MAX_MOVEMENT = 80.0


with open(METADATA_FILE, "r") as file:
    metadata = json.load(file)

if metadata.get("feature_version") != FEATURE_VERSION:
    raise RuntimeError(
        "Model feature version does not match "
        "gaze_features.py."
    )

model = joblib.load(MODEL_FILE)
use_head_pose = metadata.get(
    "use_head_pose",
    False,
)
affine_correction = np.asarray(
    metadata.get("affine_correction", []),
    dtype=float,
)
if affine_correction.shape != (3, 2):
    affine_correction = None


root = tk.Tk()
root.withdraw()

SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

root.destroy()


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH,
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    output_facial_transformation_matrixes=use_head_pose,
)

detector = vision.FaceLandmarker.create_from_options(
    options
)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    detector.close()
    raise RuntimeError(
        "Could not open webcam."
    )


cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL,
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)


gaze_filter = KalmanGazeFilter(
    process_noise=KALMAN_PROCESS_NOISE,
    measurement_noise=KALMAN_MEASUREMENT_NOISE,
    dead_zone=KALMAN_DEADBAND,
    max_movement=KALMAN_MAX_MOVEMENT,
)

MIN_GAZE_CONFIDENCE = 0.35

session_start_time = time.perf_counter()
last_timestamp_ms = -1
last_valid_time = time.perf_counter()


def get_next_timestamp():
    global last_timestamp_ms

    timestamp_ms = int(
        (time.perf_counter() - session_start_time)
        * 1000
    )

    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1

    last_timestamp_ms = timestamp_ms

    return timestamp_ms


def reset_tracking():
    gaze_filter.reset()


def update_gaze(raw_x, raw_y):
    return gaze_filter.update(raw_x, raw_y)


print()
print("==============================================")
print("GAZE ESTIMATION")
print("==============================================")
print("Model:", metadata.get("model_type"))
print(
    "Calibration mean error:",
    f"{metadata.get('cv_mean_error_px', 0):.1f}px",
)
print(
    "Screen:",
    SCREEN_WIDTH,
    "x",
    SCREEN_HEIGHT,
)
print("Press Q to quit.")
print()


try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = detector.detect_for_video(
            image,
            get_next_timestamp(),
        )

        canvas = np.zeros(
            (
                SCREEN_HEIGHT,
                SCREEN_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        gaze_x = None
        gaze_y = None
        valid_prediction = False
        gaze_confidence = 0.0
        gaze_state = "UNCERTAIN"

        if result.face_landmarks:
            matrix = None

            if (
                use_head_pose
                and result.facial_transformation_matrixes
            ):
                matrix = (
                    result.facial_transformation_matrixes[0]
                )

            features = get_features(
                result.face_landmarks[0],
                matrix,
            )
            metrics = extract_eye_metrics(
                result.face_landmarks[0],
                matrix,
            )

            if features is not None and metrics is not None:
                gaze_confidence = estimate_gaze_confidence(metrics)

                if gaze_confidence >= MIN_GAZE_CONFIDENCE:
                    feature_array = np.asarray(
                        features,
                        dtype=float,
                    ).reshape(1, -1)

                    prediction = model.predict(
                        feature_array
                    )[0]

                    raw_x = float(prediction[0])
                    raw_y = float(prediction[1])

                    if affine_correction is not None:
                        corrected = np.array(
                            [raw_x, raw_y, 1.0]
                        ) @ affine_correction
                        raw_x = float(corrected[0])
                        raw_y = float(corrected[1])

                    smooth_x_value, smooth_y_value = (
                        update_gaze(raw_x, raw_y)
                    )

                    gaze_x = int(
                        np.clip(
                            smooth_x_value,
                            0,
                            SCREEN_WIDTH - 1,
                        )
                    )

                    gaze_y = int(
                        np.clip(
                            smooth_y_value,
                            0,
                            SCREEN_HEIGHT - 1,
                        )
                    )

                    valid_prediction = True
                    last_valid_time = time.perf_counter()

                    if abs(gaze_x - SCREEN_WIDTH / 2) < 0.1 * SCREEN_WIDTH and abs(gaze_y - SCREEN_HEIGHT / 2) < 0.1 * SCREEN_HEIGHT:
                        gaze_state = "CENTER"
                    elif gaze_x < SCREEN_WIDTH * 0.45:
                        gaze_state = "LEFT"
                    elif gaze_x > SCREEN_WIDTH * 0.55:
                        gaze_state = "RIGHT"
                    elif gaze_y < SCREEN_HEIGHT * 0.45:
                        gaze_state = "UP"
                    elif gaze_y > SCREEN_HEIGHT * 0.55:
                        gaze_state = "DOWN"
                    else:
                        gaze_state = "CENTER"

        if not valid_prediction:
            if (
                time.perf_counter() - last_valid_time
                > 0.5
            ):
                reset_tracking()

            cv2.putText(
                canvas,
                "Face or eyes not detected",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                canvas,
                "STATE: UNCERTAIN",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        else:
            cv2.circle(
                canvas,
                (gaze_x, gaze_y),
                24,
                (0, 255, 0),
                -1,
            )

            cv2.circle(
                canvas,
                (gaze_x, gaze_y),
                36,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                canvas,
                f"Gaze: ({gaze_x}, {gaze_y})",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"STATE: {gaze_state}",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"CONFIDENCE: {gaze_confidence:.2f}",
            (30, SCREEN_HEIGHT - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Move your eyes and keep your head still",
            (30, SCREEN_HEIGHT - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            WINDOW_NAME,
            canvas,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()