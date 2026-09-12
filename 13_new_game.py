import csv
import json
import math
import os
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_features import FEATURE_VERSION, get_features

# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================

NUM_TRIALS = 20
NUM_TARGETS = 6
MIN_FIXATION_MS = 400
TARGET_SIZE = 120
TRIAL_TIMEOUT_MS = 8000
INTER_TRIAL_DELAY = 1200
SMOOTHING_ALPHA = 0.13
DEAD_ZONE = 4.0
TARGET_ZONE_TOLERANCE = 0.0  # kept for compatibility; proximity scoring is used below
SHOW_GAZE_CURSOR = False
SHOW_TARGET_LABEL = False
SHOW_INSTRUCTION_TEXT = True
USE_HEAD_POSE = False
OUTPUT_CSV = "gaze_spot_it_results.csv"
GAZE_X_OFFSET_PX = -8
GAZE_Y_OFFSET_PX = 0
RAW_HISTORY_SIZE = 7
TRACKER_DEADBAND = 2.0
MAX_FRAME_MOVEMENT = 35

# Proximity-based concentration scoring (same idea as the first game)
# A gaze point does not need to land exactly on the target.
CONCENTRATION_RADIUS_MULTIPLIER = 3.0
CONCENTRATION_SCORE_EXPONENT = 0.65
MIN_CONCENTRATION_FOR_FIXATION = 0.20

# ==========================================================
# SCREEN / MODEL SETUP
# ==========================================================

root = tk.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

MODEL_FILE = "gaze_model_accurate.pkl" if os.path.exists("gaze_model_accurate.pkl") else "gaze_model.pkl"
METADATA_FILE = (
    "gaze_model_accurate_metadata.json"
    if os.path.exists("gaze_model_accurate_metadata.json")
    else "gaze_model_metadata.json"
)

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

if metadata.get("feature_version") != FEATURE_VERSION:
    raise RuntimeError(
        "Model and feature versions do not match. Run the calibration again."
    )

affine_correction = np.array(metadata.get("affine_correction", []), dtype=float)
if affine_correction.shape != (3, 2):
    affine_correction = None

model = joblib.load(MODEL_FILE)
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
    raise RuntimeError("Unable to open webcam.")

WINDOW_NAME = "Gaze Spot-It Prototype"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ==========================================================
# DATA STRUCTURES
# ==========================================================

@dataclass
class TargetZone:
    target_id: int
    cx: float
    cy: float
    width: int
    height: int
    x1: float
    y1: float
    x2: float
    y2: float


class GazeSmoother:
    """Mimics the more stable tracking pattern used in the concentration game."""

    def __init__(self, alpha=0.13, dead_zone=4.0):
        self.alpha = alpha
        self.dead_zone = dead_zone
        self.filtered_x = None
        self.filtered_y = None
        self.raw_history = deque(maxlen=RAW_HISTORY_SIZE)

    def update(self, raw_x, raw_y):
        if raw_x is None or raw_y is None:
            return None, None

        raw_x = float(raw_x) + GAZE_X_OFFSET_PX
        raw_y = float(raw_y) + GAZE_Y_OFFSET_PX
        self.raw_history.append((raw_x, raw_y))

        if self.filtered_x is None or self.filtered_y is None:
            self.filtered_x, self.filtered_y = raw_x, raw_y
            return self.filtered_x, self.filtered_y

        filtered_x, filtered_y = np.median(np.array(self.raw_history), axis=0)
        next_x = self.alpha * filtered_x + (1.0 - self.alpha) * self.filtered_x
        next_y = self.alpha * filtered_y + (1.0 - self.alpha) * self.filtered_y

        movement = math.hypot(next_x - self.filtered_x, next_y - self.filtered_y)
        if movement < TRACKER_DEADBAND:
            next_x, next_y = self.filtered_x, self.filtered_y
        if movement > MAX_FRAME_MOVEMENT:
            scale = MAX_FRAME_MOVEMENT / movement
            next_x = self.filtered_x + (next_x - self.filtered_x) * scale
            next_y = self.filtered_y + (next_y - self.filtered_y) * scale

        self.filtered_x, self.filtered_y = next_x, next_y
        return self.filtered_x, self.filtered_y

    def reset(self):
        self.filtered_x = None
        self.filtered_y = None
        self.raw_history.clear()


def ensure_output_file(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])


# ==========================================================
# STIMULUS GENERATION
# ==========================================================


def build_target_positions(num_targets, target_size, padding=120):
    """Generate a randomized but screen-safe distribution of target centers."""
    positions = []
    min_x = padding + target_size // 2
    max_x = SCREEN_WIDTH - padding - target_size // 2
    min_y = padding + target_size // 2
    max_y = SCREEN_HEIGHT - padding - target_size // 2

    attempts = 0
    while len(positions) < num_targets and attempts < 5000:
        attempts += 1
        cx = random.randint(min_x, max_x)
        cy = random.randint(min_y, max_y)
        ok = True
        for px, py in positions:
            if math.hypot(cx - px, cy - py) < (target_size * 1.8):
                ok = False
                break
        if ok:
            positions.append((cx, cy))

    if len(positions) < num_targets:
        # Fallback: a simple grid-based distribution that keeps the trial valid.
        cols = int(math.ceil(math.sqrt(num_targets)))
        rows = int(math.ceil(num_targets / cols))
        x_spaces = np.linspace(min_x, max_x, cols)
        y_spaces = np.linspace(min_y, max_y, rows)
        for row_idx in range(rows):
            for col_idx in range(cols):
                if len(positions) >= num_targets:
                    break
                positions.append((int(x_spaces[col_idx]), int(y_spaces[row_idx])))

    random.shuffle(positions)
    return positions


def get_target_zone(cx, cy, width, height):
    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0
    return TargetZone(-1, cx, cy, width, height, x1, y1, x2, y2)


def target_proximity_score(px, py, target):
    """
    Return a continuous 0..1 concentration score based on distance
    from the gaze point to the target center.

    1.0 = gaze at target center
    0.0 = gaze outside the generous concentration radius

    This deliberately avoids an exact-match / hard rectangle test.
    """
    if px is None or py is None:
        return 0.0

    distance = math.hypot(float(px) - target.cx, float(py) - target.cy)

    radius = max(
        target.width,
        target.height
    ) * 0.5 * CONCENTRATION_RADIUS_MULTIPLIER

    if distance >= radius:
        return 0.0

    proximity = 1.0 - (distance / radius)

    # Exponent < 1 makes nearby-but-not-perfect gaze count more strongly.
    return float(proximity ** CONCENTRATION_SCORE_EXPONENT)


def draw_face_icon(canvas, cx, cy, size, face_color=(120, 180, 255), is_target=False):
    """Target faces have a visible smile; distractors stay more neutral and blank."""
    radius = int(size / 2)
    cv2.circle(canvas, (int(cx), int(cy)), radius, face_color, -1)

    eye_y = int(cy - radius * 0.18)
    eye_offset = int(radius * 0.32)
    cv2.circle(canvas, (int(cx - eye_offset), eye_y), int(size * 0.08), (30, 30, 30), -1)
    cv2.circle(canvas, (int(cx + eye_offset), eye_y), int(size * 0.08), (30, 30, 30), -1)

    if is_target:
        cv2.ellipse(
            canvas,
            (int(cx), int(cy + radius * 0.20)),
            (int(radius * 0.45), int(radius * 0.28)),
            0,
            0,
            180,
            (40, 40, 40),
            3,
        )
        cv2.ellipse(
            canvas,
            (int(cx), int(cy + radius * 0.14)),
            (int(radius * 0.30), int(radius * 0.14)),
            0,
            180,
            360,
            (255, 120, 120),
            2,
        )
    else:
        cv2.line(canvas, (int(cx - radius * 0.30), int(cy + radius * 0.28)), (int(cx), int(cy + radius * 0.18)), (50, 50, 50), 2)
        cv2.line(canvas, (int(cx), int(cy + radius * 0.18)), (int(cx + radius * 0.30), int(cy + radius * 0.28)), (50, 50, 50), 2)

    if is_target and SHOW_TARGET_LABEL:
        cv2.putText(
            canvas,
            "TARGET",
            (int(cx - 28), int(cy + radius + 34)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )


def draw_background(canvas):
    h, w = canvas.shape[:2]

    # Soft pastel sky that keeps the center area clear for the target layout.
    cv2.rectangle(canvas, (0, 0), (w, h), (214, 230, 255), -1)
    cv2.rectangle(canvas, (0, int(h * 0.72)), (w, h), (164, 224, 171), -1)

    # Gentle hills / ground to separate the faces from the background.
    cv2.ellipse(canvas, (int(w * 0.25), int(h * 0.82)), (int(w * 0.28), int(h * 0.12)), 0, 0, 360, (150, 205, 150), -1)
    cv2.ellipse(canvas, (int(w * 0.72), int(h * 0.80)), (int(w * 0.32), int(h * 0.12)), 0, 0, 360, (135, 198, 140), -1)

    # Simple cloud shapes to feel playful without competing with the face targets.
    cloud_positions = [(int(w * 0.14), int(h * 0.18)), (int(w * 0.55), int(h * 0.16)), (int(w * 0.82), int(h * 0.20))]
    for cx, cy in cloud_positions:
        cv2.ellipse(canvas, (cx, cy), (80, 26), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, (cx - 48, cy + 10), (52, 20), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(canvas, (cx + 52, cy + 12), (52, 20), 0, 0, 360, (255, 255, 255), -1)

    # Gentle ground texture that helps the face icons remain clearly distinct.
    for x in range(40, w, 80):
        cv2.circle(canvas, (x, int(h * 0.86)), 7, (255, 255, 255), -1)
        cv2.circle(canvas, (x + 18, int(h * 0.90)), 7, (255, 255, 255), -1)


def draw_scene(canvas, target_zones, selected_index, gaze_x=None, gaze_y=None):
    draw_background(canvas)

    for idx, zone in enumerate(target_zones):
        color = (180, 200, 220)
        if idx == selected_index:
            color = (120, 200, 130)
        draw_face_icon(canvas, zone.cx, zone.cy, TARGET_SIZE, face_color=color, is_target=(idx == selected_index))

    cv2.putText(
        canvas,
        "Find the smiling face",
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
    )

    if SHOW_INSTRUCTION_TEXT:
        cv2.putText(
            canvas,
            f"Trial {selected_index + 1} / {len(target_zones)}",
            (40, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )


# ==========================================================
# GAZE PROCESSING
# ==========================================================


def process_gaze(frame, detector, timestamp_ms, model, use_head_pose=False):
    """Return raw gaze coordinates plus smoothed gaze values, or None if invalid."""
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    raw_x = None
    raw_y = None
    confidence = None

    if result.face_landmarks:
        matrix = (
            result.facial_transformation_matrixes[0]
            if (use_head_pose and result.facial_transformation_matrixes)
            else None
        )
        features = get_features(result.face_landmarks[0], matrix)
        if features is not None:
            pred = model.predict(np.array(features).reshape(1, -1))[0]
            raw_x, raw_y = float(pred[0]), float(pred[1])

            if affine_correction is not None:
                corrected = np.array([raw_x, raw_y, 1.0]) @ affine_correction
                raw_x, raw_y = float(corrected[0]), float(corrected[1])

            # Optional confidence signal if the model or face tracker exposes it. Here it is not available,
            # so we leave confidence as None and log blank values in the CSV.
            confidence = None

    return result, raw_x, raw_y, confidence


# ==========================================================
# TRIAL LOGGING
# ==========================================================


def log_summary(results):
    if not results:
        print("No trials were completed.")
        return

    latencies = [r["target_acquisition_latency_ms"] for r in results if r["successful_acquisition"]]
    fixation_ms = [r["fixation_duration_ms"] for r in results if r["successful_acquisition"]]
    variability = [r["gaze_variability_x"] for r in results if r["gaze_variability_x"] is not None]
    distractor_total = sum(int(r["number_of_distractor_entries"]) for r in results)
    concentration_values = [
        r["mean_target_concentration"]
        for r in results
        if r.get("mean_target_concentration") is not None
    ]

    success_rate = (sum(1 for r in results if r["successful_acquisition"]) / len(results)) * 100.0
    mean_latency = statistics.mean(latencies) if latencies else 0.0
    median_latency = statistics.median(latencies) if latencies else 0.0
    mean_fixation = statistics.mean(fixation_ms) if fixation_ms else 0.0
    mean_variability = statistics.mean(variability) if variability else 0.0
    mean_concentration = statistics.mean(concentration_values) if concentration_values else 0.0

    print("\n====================================")
    print("SESSION SUMMARY")
    print("====================================")
    print(f"Trials completed: {len(results)}")
    print(f"Successful target acquisitions: {sum(1 for r in results if r['successful_acquisition'])}")
    print(f"Mean acquisition latency: {mean_latency:.1f} ms")
    print(f"Median acquisition latency: {median_latency:.1f} ms")
    print(f"Mean fixation duration: {mean_fixation:.1f} ms")
    print(f"Target acquisition success rate: {success_rate:.1f}%")
    print(f"Mean gaze variability: {mean_variability:.2f} px")
    print(f"Distractor fixation count: {distractor_total}")
    print(f"Mean target concentration: {mean_concentration * 100:.1f}%")


# ==========================================================
# MAIN EXPERIMENT LOOP
# ==========================================================


def run_experiment():
    timestamp_ms = 0
    results = []
    smoother = GazeSmoother(alpha=SMOOTHING_ALPHA, dead_zone=DEAD_ZONE)

    output_exists = os.path.exists(OUTPUT_CSV)
    if output_exists:
        print(f"Using existing data file: {OUTPUT_CSV}")

    csv_path = OUTPUT_CSV
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "trial_id",
            "target_position_x",
            "target_position_y",
            "target_width",
            "target_height",
            "target_appearance_timestamp",
            "first_gaze_entry_timestamp",
            "target_acquisition_latency_ms",
            "fixation_start_timestamp",
            "fixation_duration_ms",
            "successful_acquisition",
            "trial_timeout",
            "number_of_target_entries",
            "number_of_distractor_entries",
            "total_gaze_samples",
            "valid_gaze_samples",
            "invalid_gaze_samples",
            "mean_gaze_x",
            "mean_gaze_y",
            "gaze_variability_x",
            "gaze_variability_y",
            "mean_confidence",
            "mean_target_concentration",
            "max_target_concentration",
            "concentration_radius_px",
        ])

        for trial_id in range(1, NUM_TRIALS + 1):
            positions = build_target_positions(NUM_TARGETS, TARGET_SIZE, padding=100)
            target_index = random.randrange(0, len(positions))
            target_cx, target_cy = positions[target_index]
            selected_target = get_target_zone(target_cx, target_cy, TARGET_SIZE, TARGET_SIZE)
            target_zone_map = []
            for i, (cx, cy) in enumerate(positions):
                zone = get_target_zone(cx, cy, TARGET_SIZE, TARGET_SIZE)
                zone.target_id = i
                target_zone_map.append(zone)

            target_appearance_ms = time.perf_counter() * 1000.0
            target_acquisition_latency_ms = None
            fixation_start_timestamp = None
            fixation_duration_ms = 0
            first_gaze_entry_timestamp = None
            successful_acquisition = False
            trial_timeout = False

            target_entry_count = 0
            distractor_entry_count = 0
            total_gaze_samples = 0
            valid_gaze_samples = 0
            invalid_gaze_samples = 0
            gaze_x_values = []
            gaze_y_values = []
            confidence_values = []
            inside_target_zone_for_ms = 0.0
            concentration_score = 0.0
            mean_target_proximity_sum = 0.0
            max_target_proximity = 0.0
            concentration_samples = 0

            gaze_x = None
            gaze_y = None
            last_visible_ms = None
            smoother.reset()

            while True:
                elapsed_ms = time.perf_counter() * 1000.0 - target_appearance_ms
                if elapsed_ms >= TRIAL_TIMEOUT_MS:
                    trial_timeout = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                result, raw_x, raw_y, confidence = process_gaze(frame, detector, timestamp_ms, model, USE_HEAD_POSE)
                timestamp_ms += 33
                total_gaze_samples += 1

                if raw_x is None or raw_y is None:
                    invalid_gaze_samples += 1
                    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                    draw_scene(canvas, target_zone_map, target_index, gaze_x, gaze_y)
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return results
                    continue

                valid_gaze_samples += 1
                gaze_x, gaze_y = smoother.update(raw_x, raw_y)
                gaze_x_values.append(float(gaze_x))
                gaze_y_values.append(float(gaze_y))
                if confidence is not None:
                    confidence_values.append(float(confidence))

                # ------------------------------------------------------
                # Proximity-based concentration
                # ------------------------------------------------------
                target_score = target_proximity_score(
                    gaze_x, gaze_y, selected_target
                )

                # Distractor proximity is also continuous. We count a
                # distractor entry only when the gaze is reasonably close
                # to another target, rather than requiring an exact hit.
                distractor_scores = [
                    target_proximity_score(gaze_x, gaze_y, zone)
                    for idx, zone in enumerate(target_zone_map)
                    if idx != target_index
                ]

                strongest_distractor_score = (
                    max(distractor_scores) if distractor_scores else 0.0
                )

                in_selected = target_score >= MIN_CONCENTRATION_FOR_FIXATION

                if in_selected:
                    if not fixation_armed:
                        fixation_armed = True
                        if first_gaze_entry_timestamp is None:
                            first_gaze_entry_timestamp = time.perf_counter() * 1000.0
                        target_entry_count += 1

                    # Instead of binary inside/outside timing, nearby gaze
                    # contributes proportionally to fixation.
                    inside_target_zone_for_ms += 33.0 * target_score
                    mean_target_proximity_sum += target_score
                    concentration_samples += 1
                    max_target_proximity = max(max_target_proximity, target_score)
                else:
                    fixation_armed = False
                    inside_target_zone_for_ms = 0.0

                if strongest_distractor_score >= MIN_CONCENTRATION_FOR_FIXATION:
                    distractor_entry_count += 1

                # 400 ms of WEIGHTED fixation is enough for acquisition.
                if in_selected and inside_target_zone_for_ms >= MIN_FIXATION_MS:
                    if not successful_acquisition:
                        fixation_start_timestamp = first_gaze_entry_timestamp
                        fixation_duration_ms = int(inside_target_zone_for_ms)
                        target_acquisition_latency_ms = int(
                            first_gaze_entry_timestamp - target_appearance_ms
                        )
                        successful_acquisition = True

                canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                draw_scene(canvas, target_zone_map, target_index, gaze_x, gaze_y)

                # Debug/status overlay
                if successful_acquisition:
                    cv2.putText(
                        canvas,
                        "Target acquired!",
                        (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        canvas,
                        f"Concentration: {target_score * 100:.0f}%  |  Weighted fixation: {int(inside_target_zone_for_ms)} ms / {MIN_FIXATION_MS} ms",
                        (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return results

                if successful_acquisition:
                    break

            # Finalize current trial metrics.
            if not gaze_x_values:
                mean_gaze_x = None
                mean_gaze_y = None
                gaze_variability_x = None
                gaze_variability_y = None
                mean_confidence = None
            else:
                mean_gaze_x = float(np.mean(gaze_x_values))
                mean_gaze_y = float(np.mean(gaze_y_values))
                gaze_variability_x = float(np.std(gaze_x_values))
                gaze_variability_y = float(np.std(gaze_y_values))
                mean_confidence = float(np.mean(confidence_values)) if confidence_values else None

            if target_acquisition_latency_ms is None and first_gaze_entry_timestamp is not None:
                target_acquisition_latency_ms = int(first_gaze_entry_timestamp - target_appearance_ms)

            if not successful_acquisition and trial_timeout:
                fixation_duration_ms = int(inside_target_zone_for_ms)
                if fixation_duration_ms > 0 and first_gaze_entry_timestamp is not None:
                    target_acquisition_latency_ms = int(first_gaze_entry_timestamp - target_appearance_ms)

            mean_target_concentration = (
                mean_target_proximity_sum / concentration_samples
                if concentration_samples else 0.0
            )
            concentration_radius_px = (
                max(TARGET_SIZE, TARGET_SIZE) * 0.5 * CONCENTRATION_RADIUS_MULTIPLIER
            )

            row = [
                trial_id,
                int(target_cx),
                int(target_cy),
                int(TARGET_SIZE),
                int(TARGET_SIZE),
                int(target_appearance_ms),
                None if first_gaze_entry_timestamp is None else int(first_gaze_entry_timestamp),
                None if target_acquisition_latency_ms is None else int(target_acquisition_latency_ms),
                None if fixation_start_timestamp is None else int(fixation_start_timestamp),
                int(fixation_duration_ms),
                int(successful_acquisition),
                int(TRIAL_TIMEOUT_MS),
                int(target_entry_count),
                int(distractor_entry_count),
                int(total_gaze_samples),
                int(valid_gaze_samples),
                int(invalid_gaze_samples),
                None if mean_gaze_x is None else round(mean_gaze_x, 2),
                None if mean_gaze_y is None else round(mean_gaze_y, 2),
                None if gaze_variability_x is None else round(gaze_variability_x, 2),
                None if gaze_variability_y is None else round(gaze_variability_y, 2),
                None if mean_confidence is None else round(mean_confidence, 3),
                round(mean_target_concentration, 3),
                round(max_target_proximity, 3),
                round(concentration_radius_px, 1),
            ]
            writer.writerow(row)

            results.append({
                "trial_id": trial_id,
                "target_position_x": int(target_cx),
                "target_position_y": int(target_cy),
                "target_appearance_timestamp": int(target_appearance_ms),
                "first_gaze_entry_timestamp": None if first_gaze_entry_timestamp is None else int(first_gaze_entry_timestamp),
                "target_acquisition_latency_ms": None if target_acquisition_latency_ms is None else int(target_acquisition_latency_ms),
                "fixation_start_timestamp": None if fixation_start_timestamp is None else int(fixation_start_timestamp),
                "fixation_duration_ms": int(fixation_duration_ms),
                "successful_acquisition": bool(successful_acquisition),
                "trial_timeout": bool(trial_timeout),
                "number_of_target_entries": int(target_entry_count),
                "number_of_distractor_entries": int(distractor_entry_count),
                "total_gaze_samples": int(total_gaze_samples),
                "valid_gaze_samples": int(valid_gaze_samples),
                "invalid_gaze_samples": int(invalid_gaze_samples),
                "mean_gaze_x": mean_gaze_x,
                "mean_gaze_y": mean_gaze_y,
                "gaze_variability_x": gaze_variability_x,
                "gaze_variability_y": gaze_variability_y,
                "mean_confidence": mean_confidence,
                "mean_target_concentration": mean_target_concentration,
                "max_target_concentration": max_target_proximity,
                "concentration_radius_px": concentration_radius_px,
            })

            if not successful_acquisition:
                canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    canvas,
                    "Next trial starting...",
                    (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(INTER_TRIAL_DELAY)

    log_summary(results)
    print(f"Trial data saved to: {OUTPUT_CSV}")
    return results


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    print("Starting gaze spot-it prototype...")
    print(f"Screen size: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
    print(f"Trials: {NUM_TRIALS}, targets per layout: {NUM_TARGETS}, fixation threshold: {MIN_FIXATION_MS} ms")
    print("Instructions: look at the smiling face and maintain attention.")
    print(
        f"Proximity scoring: {CONCENTRATION_RADIUS_MULTIPLIER:.1f}x target radius, "
        f"exponent={CONCENTRATION_SCORE_EXPONENT}"
    )

    try:
        run_experiment()
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()