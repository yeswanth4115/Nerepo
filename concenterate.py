import json
import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk
import time
import csv
import math
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gaze_features import FEATURE_VERSION, get_features
from gaze_kalman import KalmanGazeFilter


# ==========================================================
# 1. LOAD GAZE MODEL
# ==========================================================

with open("gaze_model_metadata.json", "r") as f:
    metadata = json.load(f)


if metadata.get("feature_version") != FEATURE_VERSION:
    raise RuntimeError(
        "Model and feature versions do not match.\n"
        "Please run 04_calibration.py again."
    )


model = joblib.load("gaze_model.pkl")

USE_HEAD_POSE = metadata.get("use_head_pose", False)


# ==========================================================
# 2. AFFINE CORRECTION
# ==========================================================

affine_correction = np.array(
    metadata.get("affine_correction", []),
    dtype=float
)

if affine_correction.shape != (3, 2):
    affine_correction = None


# ==========================================================
# 3. SCREEN SIZE
# ==========================================================

root = tk.Tk()
root.withdraw()

SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

root.destroy()


# ==========================================================
# 4. MEDIAPIPE FACE LANDMARKER
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
# 5. CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    detector.close()
    exit("ERROR: Could not open webcam.")


# ==========================================================
# 6. GAME WINDOW
# ==========================================================

WINDOW_NAME = "Visual Tracking Game"

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
# 7. TARGET SETTINGS
# ==========================================================

target_x = SCREEN_WIDTH * 0.20
target_y = SCREEN_HEIGHT * 0.50

target_radius = 70


# ==========================================================
# 8. SMOOTH TARGET MOVEMENT
# ==========================================================
#
# Instead of sudden bouncing movement, we use a sinusoidal
# movement. This is better for measuring smooth pursuit.
#

TARGET_SPEED = 0.0025

target_time = 0.0


# ==========================================================
# 9. GAZE FILTERING
# ==========================================================

gaze_filter = KalmanGazeFilter(
    process_noise=800.0,
    measurement_noise=225.0,
)


# ==========================================================
# 10. VELOCITY VARIABLES
# ==========================================================

previous_gaze_x = None
previous_gaze_y = None

previous_target_x = None
previous_target_y = None

previous_time = None


gaze_velocity = 0.0
target_velocity = 0.0

pursuit_gain = 0.0


# ==========================================================
# 11. PURSUIT GAIN SETTINGS
# ==========================================================

MIN_TARGET_VELOCITY = 20.0

MAX_REASONABLE_GAIN = 3.0

gain_history = deque(maxlen=30)


# ==========================================================
# 12. CATCH-UP MOVEMENT / SACCADE SETTINGS
# ==========================================================

SACCADE_VELOCITY_THRESHOLD = 500.0

SACCADE_MIN_INTERVAL = 0.25

last_saccade_time = -10

catchup_count = 0


# ==========================================================
# 13. BLINK DETECTION
# ==========================================================

# MediaPipe face landmark indices
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263


EAR_THRESHOLD = 0.18

eye_closed = False

blink_count = 0

last_blink_time = -10

BLINK_MIN_INTERVAL = 0.20


# ==========================================================
# 14. SESSION METRICS
# ==========================================================

session_start = time.time()

total_frames = 0

valid_gaze_frames = 0

distance_sum = 0.0

gain_sum = 0.0

valid_gain_frames = 0


# ==========================================================
# 15. CSV FILE
# ==========================================================

csv_file = open(
    "visual_tracking_session.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Time_s",
    "Target_X",
    "Target_Y",
    "Gaze_X",
    "Gaze_Y",
    "Gaze_Velocity_px_s",
    "Target_Velocity_px_s",
    "Pursuit_Gain",
    "Gaze_Target_Distance_px",
    "Catchup_Movement",
    "Blink"
])


# ==========================================================
# 16. HELPER FUNCTION
# ==========================================================

def distance_between(x1, y1, x2, y2):

    return math.hypot(
        x2 - x1,
        y2 - y1
    )


# ==========================================================
# 17. EYE ASPECT RATIO
# ==========================================================

def calculate_eye_ratio(landmarks):

    left_vertical = distance_between(
        landmarks[LEFT_EYE_TOP].x,
        landmarks[LEFT_EYE_TOP].y,
        landmarks[LEFT_EYE_BOTTOM].x,
        landmarks[LEFT_EYE_BOTTOM].y
    )

    left_horizontal = distance_between(
        landmarks[LEFT_EYE_LEFT].x,
        landmarks[LEFT_EYE_LEFT].y,
        landmarks[LEFT_EYE_RIGHT].x,
        landmarks[LEFT_EYE_RIGHT].y
    )


    right_vertical = distance_between(
        landmarks[RIGHT_EYE_TOP].x,
        landmarks[RIGHT_EYE_TOP].y,
        landmarks[RIGHT_EYE_BOTTOM].x,
        landmarks[RIGHT_EYE_BOTTOM].y
    )

    right_horizontal = distance_between(
        landmarks[RIGHT_EYE_LEFT].x,
        landmarks[RIGHT_EYE_LEFT].y,
        landmarks[RIGHT_EYE_RIGHT].x,
        landmarks[RIGHT_EYE_RIGHT].y
    )


    if left_horizontal == 0 or right_horizontal == 0:
        return 1.0


    left_ratio = (
        left_vertical /
        left_horizontal
    )

    right_ratio = (
        right_vertical /
        right_horizontal
    )


    return (
        left_ratio +
        right_ratio
    ) / 2.0


# ==========================================================
# 18. START MESSAGE
# ==========================================================

print()
print("==============================================")
print("VISUAL TRACKING GAME")
print("==============================================")

print(
    f"Model: {metadata['model_type']}"
)

print(
    f"Calibration CV Error: "
    f"{metadata['cv_mean_error_px']:.1f}px"
)

print()
print("The target will move smoothly.")
print("Ask the child to follow the target with their eyes.")
print()
print("Press Q to quit.")
print()


# ==========================================================
# 19. MAIN LOOP
# ==========================================================

timestamp_ms = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    total_frames += 1


    # ------------------------------------------------------
    # Mirror camera
    # ------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    timestamp_ms += 33


    # ------------------------------------------------------
    # Create black game canvas
    # ------------------------------------------------------

    canvas = np.zeros(
        (
            SCREEN_HEIGHT,
            SCREEN_WIDTH,
            3
        ),
        dtype=np.uint8
    )


    # ======================================================
    # 20. SMOOTH TARGET MOVEMENT
    # ======================================================

    current_time = time.time()

    elapsed = current_time - session_start

    target_time = elapsed


    # Horizontal sinusoidal movement
    target_x = (
        SCREEN_WIDTH * 0.50
        +
        SCREEN_WIDTH * 0.35
        *
        math.sin(
            target_time * TARGET_SPEED * 1000
        )
    )


    # Vertical sinusoidal movement
    target_y = (
        SCREEN_HEIGHT * 0.50
        +
        SCREEN_HEIGHT * 0.30
        *
        math.sin(
            target_time * TARGET_SPEED * 700
        )
    )


    target_x = float(
        np.clip(
            target_x,
            target_radius,
            SCREEN_WIDTH - target_radius
        )
    )


    target_y = float(
        np.clip(
            target_y,
            target_radius,
            SCREEN_HEIGHT - target_radius
        )
    )


    # ======================================================
    # 21. TARGET VELOCITY
    # ======================================================

    current_timestamp = time.time()


    if (
        previous_target_x is not None
        and previous_target_y is not None
        and previous_time is not None
    ):

        dt = (
            current_timestamp -
            previous_time
        )


        if dt > 0:

            target_velocity = (
                distance_between(
                    previous_target_x,
                    previous_target_y,
                    target_x,
                    target_y
                )
                / dt
            )


    previous_target_x = target_x
    previous_target_y = target_y


    # ======================================================
    # 22. DEFAULT VALUES
    # ======================================================

    predicted_x = None
    predicted_y = None

    gaze_distance = 0.0

    catchup_movement = 0

    blink_detected = 0


    # ======================================================
    # 23. FACE DETECTION
    # ======================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # --------------------------------------------------
        # Gaze features
        # --------------------------------------------------

        matrix = (
            result.facial_transformation_matrixes[0]
            if (
                USE_HEAD_POSE
                and result.facial_transformation_matrixes
            )
            else None
        )


        features = get_features(
            landmarks,
            matrix
        )


        if features is not None:

            feat_arr = np.array(
                features
            ).reshape(
                1,
                -1
            )


            # ==============================================
            # 24. GAZE PREDICTION
            # ==============================================

            raw_x, raw_y = model.predict(
                feat_arr
            )[0]


            # ==============================================
            # 25. AFFINE CORRECTION
            # ==============================================

            if affine_correction is not None:

                corrected = (
                    np.array([
                        raw_x,
                        raw_y,
                        1.0
                    ])
                    @ affine_correction
                )


                raw_x = corrected[0]
                raw_y = corrected[1]


            # ==============================================
            # 26. KALMAN FILTER
            # ==============================================

            filtered_x, filtered_y = gaze_filter.update(
                raw_x,
                raw_y,
            )


            # ==============================================
            # 27. GAZE VELOCITY
            # ==============================================
            #
            # IMPORTANT:
            # Velocity is calculated BEFORE heavy display
            # smoothing.
            #

            if (
                previous_gaze_x is not None
                and previous_gaze_y is not None
                and previous_time is not None
            ):

                dt = (
                    current_timestamp -
                    previous_time
                )


                if dt > 0:

                    gaze_velocity = (
                        distance_between(
                            previous_gaze_x,
                            previous_gaze_y,
                            filtered_x,
                            filtered_y
                        )
                        / dt
                    )


            previous_gaze_x = filtered_x
            previous_gaze_y = filtered_y


            # ==============================================
            # 29. SCREEN COORDINATES
            # ==============================================

            predicted_x = int(
                np.clip(
                    filtered_x,
                    0,
                    SCREEN_WIDTH - 1
                )
            )


            predicted_y = int(
                np.clip(
                    filtered_y,
                    0,
                    SCREEN_HEIGHT - 1
                )
            )


            valid_gaze_frames += 1


            # ==================================================
            # 30. GAZE-TARGET DISTANCE
            # ==================================================

            gaze_distance = distance_between(
                predicted_x,
                predicted_y,
                target_x,
                target_y
            )


            distance_sum += gaze_distance


            # ==================================================
            # 31. SMOOTH PURSUIT GAIN
            # ==================================================

            if target_velocity > MIN_TARGET_VELOCITY:

                pursuit_gain = (
                    gaze_velocity /
                    target_velocity
                )


                # Ignore physically unreasonable values
                # caused by tracking noise.

                if (
                    0 <= pursuit_gain
                    <= MAX_REASONABLE_GAIN
                ):

                    gain_history.append(
                        pursuit_gain
                    )

                    gain_sum += pursuit_gain

                    valid_gain_frames += 1

            else:

                pursuit_gain = 0.0


            # ==================================================
            # 32. CATCH-UP MOVEMENT
            # ==================================================

            catchup_movement = 0


            if gaze_velocity > SACCADE_VELOCITY_THRESHOLD:

                if (
                    current_timestamp -
                    last_saccade_time
                    >
                    SACCADE_MIN_INTERVAL
                ):

                    catchup_count += 1

                    catchup_movement = 1

                    last_saccade_time = (
                        current_timestamp
                    )


            # ==================================================
            # 33. BLINK DETECTION
            # ==================================================

            eye_ratio = calculate_eye_ratio(
                landmarks
            )


            if eye_ratio < EAR_THRESHOLD:

                if not eye_closed:

                    eye_closed = True

            else:

                if eye_closed:

                    if (
                        current_timestamp -
                        last_blink_time
                        >
                        BLINK_MIN_INTERVAL
                    ):

                        blink_count += 1

                        blink_detected = 1

                        last_blink_time = (
                            current_timestamp
                        )

                    eye_closed = False


    # ======================================================
    # 34. SAVE DATA
    # ======================================================

    csv_writer.writerow([
        round(
            elapsed,
            3
        ),

        round(
            target_x,
            2
        ),

        round(
            target_y,
            2
        ),

        (
            predicted_x
            if predicted_x is not None
            else ""
        ),

        (
            predicted_y
            if predicted_y is not None
            else ""
        ),

        round(
            gaze_velocity,
            2
        ),

        round(
            target_velocity,
            2
        ),

        round(
            pursuit_gain,
            3
        ),

        round(
            gaze_distance,
            2
        ),

        catchup_movement,

        blink_detected
    ])


    # ======================================================
    # 35. DRAW TARGET
    # ======================================================

    target_color = (
        0,
        255,
        255
    )


    cv2.circle(
        canvas,
        (
            int(target_x),
            int(target_y)
        ),
        target_radius,
        target_color,
        -1
    )


    # Target center
    cv2.circle(
        canvas,
        (
            int(target_x),
            int(target_y)
        ),
        10,
        (
            0,
            0,
            255
        ),
        -1
    )


    # ======================================================
    # 36. DISPLAY GAZE MARKER
    # ======================================================

    if predicted_x is not None:

        cv2.circle(
            canvas,
            (
                predicted_x,
                predicted_y
            ),
            12,
            (
                0,
                255,
                0
            ),
            -1
        )


    # ======================================================
    # 37. LIVE METRICS
    # ======================================================

    if valid_gain_frames > 0:

        average_gain = (
            gain_sum /
            valid_gain_frames
        )

    else:

        average_gain = 0.0


    if valid_gaze_frames > 0:

        average_distance = (
            distance_sum /
            valid_gaze_frames
        )

    else:

        average_distance = 0.0


    session_minutes = (
        elapsed / 60.0
    )


    if session_minutes > 0:

        blink_rate = (
            blink_count /
            session_minutes
        )

    else:

        blink_rate = 0.0


    # ======================================================
    # 38. TRACKING QUALITY
    # ======================================================

    if average_gain >= 0.75:

        tracking_quality = "GOOD"

    elif average_gain >= 0.45:

        tracking_quality = "MODERATE"

    else:

        tracking_quality = "LOW"


    # ======================================================
    # 39. DISPLAY INFORMATION
    # ======================================================

    cv2.putText(
        canvas,
        "VISUAL TRACKING GAME",
        (
            30,
            50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        canvas,
        f"Pursuit Gain: {average_gain:.2f}",
        (
            30,
            95
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        canvas,
        f"Catch-up Movements: {catchup_count}",
        (
            30,
            135
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        canvas,
        f"Blink Rate: {blink_rate:.1f}/min",
        (
            30,
            175
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        canvas,
        f"Mean Gaze Error: {average_distance:.0f}px",
        (
            30,
            215
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        canvas,
        f"Tracking: {tracking_quality}",
        (
            30,
            255
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (
            255,
            255,
            255
        ),
        2
    )


    # ======================================================
    # 40. INSTRUCTION
    # ======================================================

    cv2.putText(
        canvas,
        "Follow the moving target with your eyes",
        (
            30,
            SCREEN_HEIGHT - 50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (
            255,
            255,
            255
        ),
        2
    )


    # ======================================================
    # 41. SHOW
    # ======================================================

    cv2.imshow(
        WINDOW_NAME,
        canvas
    )


    # ======================================================
    # 42. QUIT
    # ======================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    previous_time = current_timestamp


# ==========================================================
# 43. CLEANUP
# ==========================================================

cap.release()

detector.close()

csv_file.close()

cv2.destroyAllWindows()


# ==========================================================
# 44. FINAL SESSION METRICS
# ==========================================================

session_duration = (
    time.time() -
    session_start
)


if valid_gain_frames > 0:

    final_gain = (
        gain_sum /
        valid_gain_frames
    )

else:

    final_gain = 0.0


if valid_gaze_frames > 0:

    final_mean_distance = (
        distance_sum /
        valid_gaze_frames
    )

else:

    final_mean_distance = 0.0


duration_minutes = (
    session_duration /
    60.0
)


if duration_minutes > 0:

    final_blink_rate = (
        blink_count /
        duration_minutes
    )

else:

    final_blink_rate = 0.0


# ==========================================================
# 45. FINAL REPORT
# ==========================================================

print()
print("==============================================")
print("VISUAL TRACKING SESSION SUMMARY")
print("==============================================")

print(
    f"Session Duration:       "
    f"{session_duration:.1f} seconds"
)

print(
    f"Valid Gaze Frames:      "
    f"{valid_gaze_frames}"
)

print(
    f"Smooth Pursuit Gain:    "
    f"{final_gain:.3f}"
)

print(
    f"Catch-up Movements:     "
    f"{catchup_count}"
)

print(
    f"Blink Count:            "
    f"{blink_count}"
)

print(
    f"Blink Rate:             "
    f"{final_blink_rate:.2f} / minute"
)

print(
    f"Mean Gaze Error:        "
    f"{final_mean_distance:.2f} pixels"
)

print()
print(
    "Session data saved to:"
)

print(
    "visual_tracking_session.csv"
)

print("==============================================")