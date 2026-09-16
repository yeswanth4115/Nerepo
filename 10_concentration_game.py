import csv
import json
import math
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_features import FEATURE_VERSION, get_features


# ==========================================================
# 1. SCREEN SIZE
# ==========================================================

root = tk.Tk()
root.withdraw()

SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

root.destroy()

print("Screen:", SCREEN_WIDTH, "x", SCREEN_HEIGHT)


# ==========================================================
# 2. LOAD MODEL + METADATA
# ==========================================================

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

print()
print("==============================================")
print("GAZE MODEL LOADED")
print("Model:", metadata.get("model_type"))
print("Feature version:", metadata.get("feature_version"))
print(
    "Calibration error:",
    f"{metadata.get('cv_mean_error_px', 0):.1f}px"
)
print("==============================================")


# ==========================================================
# 3. MEDIAPIPE
# ==========================================================

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

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


# ==========================================================
# 4. CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    detector.close()
    raise RuntimeError("ERROR: Could not open webcam.")


# ==========================================================
# 5. WINDOW
# ==========================================================

WINDOW_NAME = "Autism Visual Attention Assessment"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ==========================================================
# 6. MOVING TARGET
# ==========================================================

target_x = SCREEN_WIDTH // 2
target_y = SCREEN_HEIGHT // 2

target_radius = 60

speed_x = 4
speed_y = 3


# ==========================================================
# 7. GAZE SMOOTHING
# ==========================================================

history = deque(maxlen=5)

smooth_x = None
smooth_y = None

ALPHA = 0.20


# ==========================================================
# 8. CONCENTRATION SETTINGS
# ==========================================================

CONCENTRATION_RADIUS = 300

total_frames = 0
concentration_sum = 0.0
distance_sum = 0.0


# ==========================================================
# 9. CSV
# ==========================================================

csv_file = open(
    "concentration_session.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame",
    "Target_X",
    "Target_Y",
    "Gaze_X",
    "Gaze_Y",
    "Distance",
    "Concentration_Percentage"
])


# ==========================================================
# 10. TIMESTAMP
# ==========================================================

start_time = time.perf_counter()

print()
print("Concentration assessment started.")
print("Follow the moving target using your eyes.")
print("Press Q to quit.")


# ==========================================================
# 11. MAIN LOOP
# ==========================================================

try:
    while True:
        # --------------------------------------------------
        # READ CAMERA
        # --------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)

        # --------------------------------------------------
        # MEDIAPIPE
        # --------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # --------------------------------------------------
        # MOVE TARGET
        # --------------------------------------------------

        target_x += speed_x
        target_y += speed_y

        # Horizontal bounce
        if (
            target_x - target_radius <= 0
            or
            target_x + target_radius >= SCREEN_WIDTH
        ):
            speed_x *= -1

        # Vertical bounce
        if (
            target_y - target_radius <= 0
            or
            target_y + target_radius >= SCREEN_HEIGHT
        ):
            speed_y *= -1

        # --------------------------------------------------
        # CANVAS
        # --------------------------------------------------

        canvas = np.zeros(
            (
                SCREEN_HEIGHT,
                SCREEN_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        # --------------------------------------------------
        # DEFAULT VALUES
        # --------------------------------------------------

        gaze_x = None
        gaze_y = None
        distance = None
        concentration = 0.0

        # --------------------------------------------------
        # FACE DETECTED
        # --------------------------------------------------

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            matrix = None

            if (
                USE_HEAD_POSE
                and result.facial_transformation_matrixes
                and len(result.facial_transformation_matrixes) > 0
            ):
                matrix = result.facial_transformation_matrixes[0]

            # --------------------------------------------------
            # EXTRACT FEATURES USING gaze_features.get_features
            # --------------------------------------------------

            features = get_features(landmarks, matrix)

            if features is not None:
                feature_array = np.asarray(
                    features,
                    dtype=float
                ).reshape(1, -1)

                # --------------------------------------------------
                # PREDICT GAZE
                # --------------------------------------------------

                prediction = model.predict(feature_array)[0]

                raw_x = float(prediction[0])
                raw_y = float(prediction[1])

                # --------------------------------------------------
                # MEDIAN FILTER
                # --------------------------------------------------

                history.append((raw_x, raw_y))

                median_xy = np.median(
                    np.asarray(history),
                    axis=0
                )

                filtered_x = float(median_xy[0])
                filtered_y = float(median_xy[1])

                # --------------------------------------------------
                # EMA FILTER
                # --------------------------------------------------

                if smooth_x is None or smooth_y is None:
                    smooth_x = filtered_x
                    smooth_y = filtered_y
                else:
                    smooth_x = ALPHA * filtered_x + (1 - ALPHA) * smooth_x
                    smooth_y = ALPHA * filtered_y + (1 - ALPHA) * smooth_y

                # --------------------------------------------------
                # SCREEN COORDINATES
                # --------------------------------------------------

                gaze_x = int(
                    np.clip(
                        smooth_x,
                        0,
                        SCREEN_WIDTH - 1
                    )
                )

                gaze_y = int(
                    np.clip(
                        smooth_y,
                        0,
                        SCREEN_HEIGHT - 1
                    )
                )

                # --------------------------------------------------
                # DISTANCE
                # --------------------------------------------------

                distance = math.hypot(
                    gaze_x - target_x,
                    gaze_y - target_y
                )

                # --------------------------------------------------
                # CONCENTRATION
                # --------------------------------------------------

                if distance <= CONCENTRATION_RADIUS:
                    proximity = (
                        1.0
                        - (distance / CONCENTRATION_RADIUS)
                    )
                    concentration = (proximity ** 0.65) * 100.0
                else:
                    concentration = 0.0

                # --------------------------------------------------
                # METRICS
                # --------------------------------------------------

                total_frames += 1
                concentration_sum += concentration
                distance_sum += distance

                # --------------------------------------------------
                # SAVE
                                # --------------------------------------------------
                # SAVE
                # --------------------------------------------------

                csv_writer.writerow([
                    total_frames,
                    target_x,
                    target_y,
                    gaze_x,
                    gaze_y,
                    distance,
                    concentration
                ])

        # --------------------------------------------------
        # DRAW TARGET
        # --------------------------------------------------

        cv2.circle(
            canvas,
            (target_x, target_y),
            target_radius,
            (0, 255, 0),
            -1
        )

        # --------------------------------------------------
        # DRAW GAZE
        # --------------------------------------------------

        if gaze_x is not None and gaze_y is not None:

            cv2.circle(
                canvas,
                (gaze_x, gaze_y),
                20,
                (255, 0, 0),
                -1
            )

            cv2.line(
                canvas,
                (target_x, target_y),
                (gaze_x, gaze_y),
                (255, 255, 255),
                2
            )

        # --------------------------------------------------
        # DISPLAY INFORMATION
        # --------------------------------------------------

        cv2.putText(
            canvas,
            f"Concentration: {concentration:.1f}%",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )

        cv2.putText(
            canvas,
            "Follow the moving target using your eyes",
            (50, SCREEN_HEIGHT - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # --------------------------------------------------
        # SHOW
        # --------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            canvas
        )

        # --------------------------------------------------
        # QUIT
        # --------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:

    cap.release()
    detector.close()
    csv_file.close()

    cv2.destroyAllWindows()

    # ------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------

    if total_frames > 0:

        average_concentration = (
            concentration_sum / total_frames
        )

        average_distance = (
            distance_sum / total_frames
        )

        print()
        print("==============================================")
        print("SESSION COMPLETE")
        print("==============================================")
        print(
            f"Frames analyzed: {total_frames}"
        )
        print(
            f"Average concentration: "
            f"{average_concentration:.2f}%"
        )
        print(
            f"Average gaze-target distance: "
            f"{average_distance:.2f}px"
        )
        print("CSV saved as: concentration_session.csv")
        print("==============================================")