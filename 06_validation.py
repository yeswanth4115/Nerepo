import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import joblib
import json
import time
import csv
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_features import get_features, FEATURE_VERSION


# ==========================================================
# 1. LOAD MODEL + METADATA, CHECK COMPATIBILITY
# ==========================================================

with open("gaze_model_metadata.json") as f:
    metadata = json.load(f)

if metadata["feature_version"] != FEATURE_VERSION:
    print("ERROR: This model was trained with feature scheme "
          f"'{metadata['feature_version']}', but this script is using "
          f"'{FEATURE_VERSION}'. Re-run calibration before validating — "
          "otherwise the numbers below are meaningless.")
    exit()

model = joblib.load("gaze_model.pkl")
USE_HEAD_POSE = metadata["use_head_pose"]

print(f"Loaded model: {metadata['model_type']}  "
      f"(calibration CV error: {metadata['cv_mean_error_px']:.1f}px)")


# ==========================================================
# 2. SCREEN SIZE
# ==========================================================

root = tk.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

if (SCREEN_WIDTH, SCREEN_HEIGHT) != (metadata["screen_width"], metadata["screen_height"]):
    print("WARNING: Screen resolution differs from calibration session "
          f"({metadata['screen_width']}x{metadata['screen_height']} -> "
          f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}). Results won't be comparable.")


# ==========================================================
# 3. MEDIAPIPE
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


def extract(result):
    if not result.face_landmarks:
        return None
    matrix = None
    if USE_HEAD_POSE and result.facial_transformation_matrixes:
        matrix = result.facial_transformation_matrixes[0]
    return get_features(result.face_landmarks[0], matrix)


# ==========================================================
# 4. HELD-OUT TEST GRID
# ==========================================================
# Offset from the calibration grid so these points were never seen
# during training — this is what makes the error number honest.
# A 4x4 grid at the midpoints between where a 5x5 calibration grid
# would have landed.

margin_x = int(SCREEN_WIDTH * 0.08)
margin_y = int(SCREEN_HEIGHT * 0.08)

cal_xs = np.linspace(margin_x, SCREEN_WIDTH - margin_x, 5)
cal_ys = np.linspace(margin_y, SCREEN_HEIGHT - margin_y, 5)

test_xs = (cal_xs[:-1] + cal_xs[1:]) / 2   # midpoints -> interior, unseen
test_ys = (cal_ys[:-1] + cal_ys[1:]) / 2

targets = [(int(x), int(y)) for y in test_ys for x in test_xs]


# ==========================================================
# 5. CAMERA + WINDOW
# ==========================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    detector.close()
    exit()

WINDOW_NAME = "Gaze Validation"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def bail_out():
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    exit()


# ==========================================================
# 6. RUN VALIDATION
# ==========================================================

results = []
timestamp_ms = 0

print()
print("==========================================")
print(f"GAZE VALIDATION  ({len(targets)} held-out points)")
print("==========================================")
time.sleep(2)

for point_idx, (target_x, target_y) in enumerate(targets):

    print(f"Testing point {point_idx + 1}/{len(targets)}")

    start = time.time()
    while time.time() - start < 0.8:
        canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        cv2.circle(canvas, (target_x, target_y), 15, (0, 0, 255), -1)
        cv2.putText(canvas, f"Point {point_idx + 1}/{len(targets)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            bail_out()

    predictions = []
    start = time.time()
    while time.time() - start < 2.0:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33

        features = extract(result)
        if features is not None:
            pred = model.predict(np.array(features).reshape(1, -1))[0]
            px = float(np.clip(pred[0], 0, SCREEN_WIDTH - 1))
            py = float(np.clip(pred[1], 0, SCREEN_HEIGHT - 1))
            predictions.append((px, py))

        canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        cv2.circle(canvas, (target_x, target_y), 15, (0, 0, 255), -1)
        if predictions:
            px, py = predictions[-1]
            cv2.circle(canvas, (int(px), int(py)), 12, (0, 255, 0), -1)
        cv2.imshow(WINDOW_NAME, canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            bail_out()

    if predictions:
        arr = np.array(predictions)
        mean_x, mean_y = arr[:, 0].mean(), arr[:, 1].mean()
        error_x = mean_x - target_x
        error_y = mean_y - target_y
        error_distance = math.hypot(error_x, error_y)

        results.append([point_idx + 1, target_x, target_y, mean_x, mean_y,
                         error_x, error_y, error_distance])

        print(f"  Target=({target_x},{target_y}) "
              f"Prediction=({mean_x:.1f},{mean_y:.1f}) "
              f"Error={error_distance:.1f}px")


# ==========================================================
# 7. SAVE + SUMMARIZE
# ==========================================================

with open("gaze_validation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Point", "Target_X", "Target_Y", "Predicted_X", "Predicted_Y",
                      "Error_X", "Error_Y", "Euclidean_Error"])
    writer.writerows(results)

if results:
    errors = [row[7] for row in results]
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    worst = max(errors)

    print()
    print("==========================================")
    print("VALIDATION COMPLETE (held-out points)")
    print("==========================================")
    print(f"Mean error:   {mean_error:.2f} px")
    print(f"Median error: {median_error:.2f} px")
    print(f"Worst error:  {worst:.2f} px")
    print(f"(Calibration-time CV estimate was {metadata['cv_mean_error_px']:.1f}px "
          "— compare these to sanity-check that CV wasn't optimistic.)")
    print()
    print("Results saved to: gaze_validation.csv")

cv2.destroyAllWindows()
cap.release()
detector.close()