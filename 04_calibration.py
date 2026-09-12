import csv
import json
import time
import tkinter as tk
import warnings
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_features import FEATURE_VERSION, feature_names, get_features

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Calibration settings
GRID_ROWS = 5
GRID_COLS = 5
SETTLE_TIME = 0.8
COLLECT_TIME = 2.5
SAMPLES_PER_POINT = 45
MAD_THRESHOLD = 3.5
USE_HEAD_POSE = False
CALIBRATION_MARGIN_X = 60
CALIBRATION_MARGIN_Y = 60

MODEL_OUTPUT = "gaze_model.pkl"
METADATA_OUTPUT = "gaze_model_metadata.json"
RAW_OUTPUT = "calibration_raw.csv"
MODEL_PATH = "models/face_landmarker.task"


def screen_size():
    root = tk.Tk()
    root.withdraw()
    size = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return size


SCREEN_WIDTH, SCREEN_HEIGHT = screen_size()

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6,
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


def draw_target(window_name, text, x, y):
    canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    cv2.circle(canvas, (x, y), 18, (0, 0, 255), -1)
    cv2.putText(
        canvas,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.imshow(window_name, canvas)


def targets_for_screen():
    xs = np.linspace(
        CALIBRATION_MARGIN_X,
        SCREEN_WIDTH - CALIBRATION_MARGIN_X,
        GRID_COLS,
    )
    ys = np.linspace(
        CALIBRATION_MARGIN_Y,
        SCREEN_HEIGHT - CALIBRATION_MARGIN_Y,
        GRID_ROWS,
    )
    return [(int(x), int(y)) for y in ys for x in xs]


def reject_outliers(point_samples):
    point_array = np.asarray(point_samples, dtype=float)
    median = np.median(point_array, axis=0)
    mad = np.median(np.abs(point_array - median), axis=0)
    mad[mad < 1e-5] = 1e-5
    keep = np.all(np.abs(point_array - median) / mad <= MAD_THRESHOLD, axis=1)
    return point_array[keep]


def collect_calibration_data():
    window_name = "Gaze Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    samples = []
    labels = []
    groups = []
    raw_frame_count = 0
    timestamp_ms = 0
    calibration_targets = targets_for_screen()

    try:
        print(f"Calibrating {len(calibration_targets)} points on {SCREEN_WIDTH}x{SCREEN_HEIGHT}.")
        print("Keep your head still and look at each red point.")
        time.sleep(2)

        for point_id, (target_x, target_y) in enumerate(calibration_targets):
            print(
                f"Point {point_id + 1}/{len(calibration_targets)}: "
                f"({target_x}, {target_y})"
            )

            settle_start = time.perf_counter()
            while time.perf_counter() - settle_start < SETTLE_TIME:
                draw_target(
                    window_name,
                    f"Point {point_id + 1}/{len(calibration_targets)}",
                    target_x,
                    target_y,
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            point_samples = []
            collect_start = time.perf_counter()
            while time.perf_counter() - collect_start < COLLECT_TIME:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(image, timestamp_ms)
                timestamp_ms += 33
                raw_frame_count += 1

                features = extract(result)
                if features is not None and np.all(np.isfinite(features)):
                    point_samples.append(features)

                draw_target(window_name, "Keep looking at the point", target_x, target_y)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

            if len(point_samples) < 8:
                print("  skipped: too few valid samples")
                continue

            kept_samples = reject_outliers(point_samples)
            if len(kept_samples) < 3:
                print("  skipped: too few samples after outlier rejection")
                continue

            # One robust vector per target prevents highly correlated video
            # frames from making the calibration appear more certain than it is.
            point_feature = np.median(kept_samples[-SAMPLES_PER_POINT:], axis=0)
            samples.append(point_feature)
            labels.append([target_x, target_y])
            groups.append(point_id)
            print(
                f"  accepted {len(kept_samples)} samples; "
                "using robust point median"
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(samples) < 15:
        raise RuntimeError(
            "Too few calibration points were collected. "
            "Improve lighting and retry."
        )

    return (
        np.asarray(samples, dtype=float),
        np.asarray(labels, dtype=float),
        np.asarray(groups),
        raw_frame_count,
        calibration_targets,
    )


def model_candidates():
    return {
        "ridge_linear": make_pipeline(StandardScaler(), Ridge(alpha=0.5)),
        "ridge_quadratic": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "svr_rbf": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(
                SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=0.01)
            ),
        ),
        "kernel_ridge_rbf": make_pipeline(
            StandardScaler(),
            KernelRidge(kernel="rbf", alpha=0.1, gamma=2.0),
        ),
    }


def evaluate_models(x, y, groups):
    logo = LeaveOneGroupOut()
    results = {}
    for name, candidate in model_candidates().items():
        errors = []
        for train_idx, test_idx in logo.split(x, y, groups):
            model = clone(candidate)
            model.fit(x[train_idx], y[train_idx])
            prediction = model.predict(x[test_idx])
            distances = np.linalg.norm(prediction - y[test_idx], axis=1)
            errors.extend(distances.tolist())

        results[name] = {
            "mean_px": float(np.mean(errors)),
            "median_px": float(np.median(errors)),
            "worst_px": float(np.max(errors)),
        }
    return results


def save_raw_data(x, y, groups):
    with open(RAW_OUTPUT, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            feature_names(USE_HEAD_POSE) + ["target_x", "target_y", "point_id"]
        )
        writer.writerows(
            [
                list(features) + [label[0], label[1], group]
                for features, label, group in zip(x, y, groups)
            ]
        )


def train_and_save(x, y, groups, raw_frame_count, target_count):
    results = evaluate_models(x, y, groups)
    print("\nPoint-held-out model comparison:")
    for name, score in results.items():
        print(
            f"  {name:18s} mean={score['mean_px']:.1f}px "
            f"median={score['median_px']:.1f}px "
            f"worst={score['worst_px']:.1f}px"
        )

    best_name = min(results, key=lambda name: results[name]["mean_px"])
    candidates = model_candidates()
    final_model = candidates[best_name]
    final_model.fit(x, y)
    joblib.dump(final_model, MODEL_OUTPUT)

    # Fit screen-space correction only from point-held-out predictions.
    logo = LeaveOneGroupOut()
    oof_predictions = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in logo.split(x, y, groups):
        fold_model = clone(candidates[best_name])
        fold_model.fit(x[train_idx], y[train_idx])
        oof_predictions[test_idx] = fold_model.predict(x[test_idx])

    affine_inputs = np.column_stack(
        [oof_predictions[:, 0], oof_predictions[:, 1], np.ones(len(x))]
    )
    affine_correction, _, _, _ = np.linalg.lstsq(affine_inputs, y, rcond=None)
    corrected_oof = affine_inputs @ affine_correction
    corrected_errors = np.linalg.norm(corrected_oof - y, axis=1)
    results[best_name]["corrected_mean_px"] = float(np.mean(corrected_errors))
    results[best_name]["corrected_median_px"] = float(np.median(corrected_errors))

    metadata = {
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names(USE_HEAD_POSE),
        "use_head_pose": USE_HEAD_POSE,
        "model_type": best_name,
        "cv_mean_error_px": results[best_name]["mean_px"],
        "cv_median_error_px": results[best_name]["median_px"],
        "cv_worst_error_px": results[best_name]["worst_px"],
        "cv_corrected_mean_error_px": results[best_name]["corrected_mean_px"],
        "cv_all_models": results,
        "affine_correction": affine_correction.tolist(),
        "screen_width": SCREEN_WIDTH,
        "screen_height": SCREEN_HEIGHT,
        "n_calibration_points": int(target_count),
        "n_accepted_points": int(len(x)),
        "n_raw_frames": int(raw_frame_count),
        "calibration_method": "robust_median_per_point",
    }
    Path(METADATA_OUTPUT).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSelected: {best_name}")
    print(f"Held-out mean error: {results[best_name]['mean_px']:.1f}px")
    print(
        "Held-out mean error after affine correction: "
        f"{results[best_name]['corrected_mean_px']:.1f}px"
    )
    print(f"Saved: {MODEL_OUTPUT}, {METADATA_OUTPUT}, {RAW_OUTPUT}")


def main():
    try:
        x, y, groups, raw_frame_count, calibration_targets = collect_calibration_data()
        save_raw_data(x, y, groups)
        train_and_save(x, y, groups, raw_frame_count, len(calibration_targets))
    finally:
        detector.close()


if __name__ == "__main__":
    main()
