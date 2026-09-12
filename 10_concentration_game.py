
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import joblib
import time
import csv
import math
import json
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gaze_features import FEATURE_VERSION, get_features

# ==========================================================
# 1. SCREEN DIMENSIONS & MODEL LOADING
# ==========================================================

root = tk.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

with open("gaze_model_metadata.json") as f:
    metadata = json.load(f)

if metadata.get("feature_version") != FEATURE_VERSION:
    raise RuntimeError(
        "Model and feature versions do not match. Run 04_calibration.py again."
    )

model = joblib.load("gaze_model.pkl")

USE_HEAD_POSE = metadata.get("use_head_pose", False)

affine_correction = np.array(
    metadata.get("affine_correction", []),
    dtype=float
)

if affine_correction.shape != (3, 2):
    affine_correction = None

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
# 2. GAME INITIALIZATION
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    detector.close()
    exit("ERROR: Camera failed to initialize.")


WINDOW_NAME = "Autism Visual Attention Assessment Game"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ==========================================================
# 3. TARGET SETTINGS
# ==========================================================

target_x = SCREEN_WIDTH // 2
target_y = SCREEN_HEIGHT // 2

target_radius = 80

speed_x = 3
speed_y = 2


# ----------------------------------------------------------
# Initial alignment
# ----------------------------------------------------------

ALIGNMENT_FRAMES = 45

alignment_samples = []

alignment_offset_x = 0.0
alignment_offset_y = 0.0

is_aligned = False


# ----------------------------------------------------------
# Gaze smoothing
# ----------------------------------------------------------

TRACKER_DEADBAND = 2
RAW_HISTORY_SIZE = 7

raw_history = deque(maxlen=RAW_HISTORY_SIZE)

smooth_x = None
smooth_y = None

alpha = 0.13

MAX_FRAME_MOVEMENT = 35


# ==========================================================
# 4. CONCENTRATION SCORING SETTINGS
# ==========================================================

# We no longer require the gaze to be exactly inside
# the target circle.
#
# Instead, gaze within this radius contributes to the score.

CONCENTRATION_RADIUS = target_radius * 3.0


# Distance at which concentration contribution becomes 0%.
#
# Example:
# target radius = 80
# concentration radius = 240
#
# At distance 0      -> 100%
# At distance 60     -> 75%
# At distance 120    -> 50%
# At distance 180    -> 25%
# At distance 240+   -> 0%

# A small minimum score can also be given to nearby gaze.
# This prevents the score from dropping too aggressively.

MIN_NEARBY_SCORE = 0.15


# ==========================================================
# 5. METRICS LOGGING
# ==========================================================

total_eval_frames = 0

# Instead of only counting "inside / outside",
# accumulate a continuous concentration value.

concentration_score_sum = 0.0

total_distance_sum = 0.0


csv_file = open(
    "concentration_session.csv",
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Timestamp_MS",
    "Target_X",
    "Target_Y",
    "Gaze_X",
    "Gaze_Y",
    "Euclidean_Distance",
    "Concentration_Score"
])


timestamp_ms = 0


# ==========================================================
# 6. MAIN ASSESSMENT LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # Mirror camera image

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    timestamp_ms += 33


    # Black fullscreen game canvas

    canvas = np.zeros(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
        dtype=np.uint8
    )


    # ======================================================
    # MOVE TARGET AFTER INITIAL ALIGNMENT
    # ======================================================

    if is_aligned:

        target_x += speed_x
        target_y += speed_y


    # Bounce from horizontal edges

    if (
        target_x - target_radius <= 0
        or
        target_x + target_radius >= SCREEN_WIDTH
    ):
        speed_x *= -1


    # Bounce from vertical edges

    if (
        target_y - target_radius <= 0
        or
        target_y + target_radius >= SCREEN_HEIGHT
    ):
        speed_y *= -1


    predicted_x = None
    predicted_y = None

    dist = 0.0

    concentration_value = 0.0


    # ======================================================
    # FACE / GAZE PROCESSING
    # ======================================================

    if result.face_landmarks:

        matrix = (
            result.facial_transformation_matrixes[0]
            if (
                USE_HEAD_POSE
                and result.facial_transformation_matrixes
            )
            else None
        )


        features = get_features(
            result.face_landmarks[0],
            matrix
        )


        if features is not None:

            feat_arr = np.array(features).reshape(1, -1)


            # ----------------------------------------------
            # Predict gaze position
            # ----------------------------------------------

            raw_x, raw_y = model.predict(feat_arr)[0]


            # ----------------------------------------------
            # Affine correction
            # ----------------------------------------------

            if affine_correction is not None:

                corrected = (
                    np.array([raw_x, raw_y, 1.0])
                    @ affine_correction
                )

                raw_x, raw_y = corrected


            # ----------------------------------------------
            # Median filtering
            # ----------------------------------------------

            raw_history.append(
                (raw_x, raw_y)
            )

            filtered_x, filtered_y = np.median(
                np.array(raw_history),
                axis=0
            )


            # ----------------------------------------------
            # EMA smoothing
            # ----------------------------------------------

            if smooth_x is None or smooth_y is None:

                smooth_x = filtered_x
                smooth_y = filtered_y

            else:

                next_x = (
                    alpha * filtered_x
                    + (1 - alpha) * smooth_x
                )

                next_y = (
                    alpha * filtered_y
                    + (1 - alpha) * smooth_y
                )


                movement = math.hypot(
                    next_x - smooth_x,
                    next_y - smooth_y
                )


                # Ignore tiny movements

                if movement < TRACKER_DEADBAND:

                    next_x = smooth_x
                    next_y = smooth_y


                # Prevent sudden jumps

                if movement > MAX_FRAME_MOVEMENT:

                    scale = (
                        MAX_FRAME_MOVEMENT
                        / movement
                    )

                    next_x = (
                        smooth_x
                        + (next_x - smooth_x) * scale
                    )

                    next_y = (
                        smooth_y
                        + (next_y - smooth_y) * scale
                    )


                smooth_x = next_x
                smooth_y = next_y


            # ==================================================
            # INITIAL PERSONAL BIAS ALIGNMENT
            # ==================================================

            predicted_x = int(
                np.clip(
                    smooth_x,
                    0,
                    SCREEN_WIDTH - 1
                )
            )

            predicted_y = int(
                np.clip(
                    smooth_y,
                    0,
                    SCREEN_HEIGHT - 1
                )
            )


            if not is_aligned:

                alignment_samples.append(
                    (smooth_x, smooth_y)
                )


                if len(alignment_samples) >= ALIGNMENT_FRAMES:

                    samples = np.array(
                        alignment_samples
                    )


                    median_x, median_y = np.median(
                        samples,
                        axis=0
                    )


                    alignment_offset_x = (
                        target_x - median_x
                    )

                    alignment_offset_y = (
                        target_y - median_y
                    )


                    is_aligned = True


            # ==================================================
            # CONCENTRATION ASSESSMENT
            # ==================================================

            if is_aligned:

                corrected_x = (
                    smooth_x
                    + alignment_offset_x
                )

                corrected_y = (
                    smooth_y
                    + alignment_offset_y
                )


                predicted_x = int(
                    np.clip(
                        corrected_x,
                        0,
                        SCREEN_WIDTH - 1
                    )
                )

                predicted_y = int(
                    np.clip(
                        corrected_y,
                        0,
                        SCREEN_HEIGHT - 1
                    )
                )


                # ----------------------------------------------
                # Calculate distance from moving target
                # ----------------------------------------------

                dist = math.hypot(
                    predicted_x - target_x,
                    predicted_y - target_y
                )


                # ==================================================
                # PROXIMITY-BASED CONCENTRATION
                # ==================================================
                #
                # No exact matching.
                #
                # The closer the gaze is to the target,
                # the greater the concentration contribution.
                #
                # 0 distance      -> 100%
                # 1/4 radius      -> high score
                # 1/2 radius      -> good score
                # target radius   -> still good
                # 2x radius       -> moderate score
                # 3x radius       -> 0%
                #

                if dist <= CONCENTRATION_RADIUS:

                    proximity = (
                        1.0
                        - (
                            dist
                            / CONCENTRATION_RADIUS
                        )
                    )


                    # Smooth the scoring curve.
                    #
                    # Squaring proximity makes nearby gaze
                    # count more strongly.

                    concentration_value = (
                        proximity ** 0.65
                    )


                    # Give nearby gaze a small minimum
                    # contribution instead of treating it
                    # as completely unfocused.

                    if dist <= target_radius * 2:

                        concentration_value = max(
                            concentration_value,
                            MIN_NEARBY_SCORE
                        )

                else:

                    concentration_value = 0.0


                # Convert to percentage

                concentration_percentage = (
                    concentration_value * 100.0
                )


                # Accumulate metrics

                total_eval_frames += 1

                concentration_score_sum += (
                    concentration_percentage
                )

                total_distance_sum += dist


                # Save session data

                csv_writer.writerow([
                    timestamp_ms,
                    target_x,
                    target_y,
                    predicted_x,
                    predicted_y,
                    round(dist, 2),
                    round(
                        concentration_percentage,
                        2
                    )
                ])


    # ======================================================
    # TARGET DISPLAY
    # ======================================================

    #
    # IMPORTANT:
    # The gaze/eye-follow marker has been completely removed.
    #
    # Only the target is displayed.
    #

    if is_aligned:

        # Make target slightly brighter when gaze is nearby.

        if (
            predicted_x is not None
            and predicted_y is not None
            and dist <= CONCENTRATION_RADIUS
        ):
            target_color = (0, 255, 0)

        else:
            target_color = (0, 0, 255)

    else:

        target_color = (0, 255, 255)


    # Main target

    cv2.circle(
        canvas,
        (target_x, target_y),
        target_radius,
        target_color,
        -1
    )


    # Target center

    cv2.circle(
        canvas,
        (target_x, target_y),
        12,
        (0, 0, 150),
        -1
    )


    # ======================================================
    # CONCENTRATION SCORE
    # ======================================================

    if total_eval_frames > 0:

        score_percentage = (
            concentration_score_sum
            / total_eval_frames
        )

    else:

        score_percentage = 0.0


    # Alignment instruction

    if not is_aligned:

        cv2.putText(
            canvas,
            "Look at the center dot",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


    # Score display

    cv2.putText(
        canvas,
        f"Concentration Score: {score_percentage:.1f}%",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )


    # ======================================================
    # DISPLAY
    # ======================================================

    cv2.imshow(
        WINDOW_NAME,
        canvas
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================================
# 7. CLEANUP
# ==========================================================

cap.release()

detector.close()

csv_file.close()

cv2.destroyAllWindows()


# ==========================================================
# 8. SESSION SUMMARY
# ==========================================================

mean_distance = (
    total_distance_sum / total_eval_frames
    if total_eval_frames > 0
    else 0
)


print("\n==========================================")
print("SESSION SUMMARY")
print("==========================================")

print(
    f"Total Evaluated Frames: {total_eval_frames}"
)

print(
    f"Mean Gaze Offset:       {mean_distance:.2f} px"
)

print(
    f"Final Concentration:    {score_percentage:.2f}%"
)

print(
    "Session data saved to: concentration_session.csv"
)

