import csv
import json
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tkinter as tk

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from gaze_features import (
    FEATURE_VERSION,
    feature_names,
    get_features,
)


GRID_ROWS = 5
GRID_COLS = 5

MARGIN_X = 100
MARGIN_Y = 80

SETTLE_TIME = 1.5
COLLECT_TIME = 3.0

MIN_SAMPLES_PER_POINT = 20
MAD_THRESHOLD = 3.5

USE_HEAD_POSE = False
MODEL_PATH = "models/face_landmarker.task"
WINDOW_NAME = "Gaze Calibration"


root = tk.Tk()
root.withdraw()

SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

root.destroy()


def generate_targets():
    x_values = np.linspace(
        MARGIN_X,
        SCREEN_WIDTH - MARGIN_X,
        GRID_COLS,
    )

    y_values = np.linspace(
        MARGIN_Y,
        SCREEN_HEIGHT - MARGIN_Y,
        GRID_ROWS,
    )

    return [
        (int(round(x)), int(round(y)))
        for y in y_values
        for x in x_values
    ]


def draw_target(x, y, point_number, total_points):
    canvas = np.zeros(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
        dtype=np.uint8,
    )

    cv2.circle(
        canvas,
        (x, y),
        28,
        (0, 0, 255),
        -1,
    )

    cv2.circle(
        canvas,
        (x, y),
        8,
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        canvas,
        "Look at the red dot",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Point {point_number}/{total_points}",
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_NAME, canvas)


def reject_outliers(samples):
    feature_count = len(
        feature_names(USE_HEAD_POSE)
    )

    values = np.asarray(
        samples,
        dtype=float,
    )

    if len(values) < MIN_SAMPLES_PER_POINT:
        return np.empty(
            (0, feature_count),
            dtype=float,
        )

    median = np.median(
        values,
        axis=0,
    )

    mad = np.median(
        np.abs(values - median),
        axis=0,
    )

    mad[mad < 0.000001] = 0.000001

    deviation = np.abs(
        values - median
    ) / mad

    keep_mask = np.all(
        deviation <= MAD_THRESHOLD,
        axis=1,
    )

    return values[keep_mask]


def calculate_error(actual, predicted):
    return np.sqrt(
        np.sum(
            (predicted - actual) ** 2,
            axis=1,
        )
    )


def evaluate_model(model, features, targets, groups):
    splitter = LeaveOneGroupOut()
    errors = []

    for train_index, test_index in splitter.split(
        features,
        targets,
        groups,
    ):
        fold_model = clone(model)

        fold_model.fit(
            features[train_index],
            targets[train_index],
        )

        prediction = fold_model.predict(
            features[test_index]
        )

        errors.extend(
            calculate_error(
                targets[test_index],
                prediction,
            ).tolist()
        )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "worst": float(np.max(errors)),
    }


targets = generate_targets()

print()
print("==============================================")
print("GAZE CALIBRATION")
print("==============================================")
print(f"Screen: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
print(f"Calibration points: {len(targets)}")
print("Keep your head still.")
print("Move only your eyes.")
print("Press Q to stop.")
print()

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
    output_facial_transformation_matrixes=USE_HEAD_POSE,
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


features_all = []
targets_all = []
groups_all = []

session_start_time = time.perf_counter()
last_timestamp_ms = -1


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


try:
    time.sleep(2)

    for point_index, (target_x, target_y) in enumerate(
        targets
    ):
        point_number = point_index + 1

        print(
            f"Collecting point "
            f"{point_number}/{len(targets)} "
            f"at ({target_x}, {target_y})"
        )

        settle_start = time.perf_counter()

        while (
            time.perf_counter() - settle_start
            < SETTLE_TIME
        ):
            draw_target(
                target_x,
                target_y,
                point_number,
                len(targets),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        point_samples = []
        collect_start = time.perf_counter()

        while (
            time.perf_counter() - collect_start
            < COLLECT_TIME
        ):
            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = detector.detect_for_video(
                image,
                get_next_timestamp(),
            )

            if result.face_landmarks:
                matrix = None

                if (
                    USE_HEAD_POSE
                    and result.facial_transformation_matrixes
                ):
                    matrix = (
                        result.facial_transformation_matrixes[0]
                    )

                values = get_features(
                    result.face_landmarks[0],
                    matrix,
                )

                if values is not None:
                    values = np.asarray(
                        values,
                        dtype=float,
                    )

                    if np.all(np.isfinite(values)):
                        point_samples.append(values)

            draw_target(
                target_x,
                target_y,
                point_number,
                len(targets),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        accepted = reject_outliers(point_samples)

        if len(accepted) < MIN_SAMPLES_PER_POINT:
            print(
                f"Rejected point: only "
                f"{len(accepted)} stable samples"
            )
            continue

        for sample in accepted:
            features_all.append(sample)
            targets_all.append(
                [target_x, target_y]
            )
            groups_all.append(point_index)

        print(
            f"Accepted {len(accepted)}/"
            f"{len(point_samples)} samples"
        )

finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


features_all = np.asarray(
    features_all,
    dtype=float,
)

targets_all = np.asarray(
    targets_all,
    dtype=float,
)

groups_all = np.asarray(
    groups_all,
    dtype=int,
)

if len(features_all) == 0:
    raise RuntimeError(
        "No calibration samples were collected."
    )

valid_points = len(
    np.unique(groups_all)
)

if valid_points < 15:
    raise RuntimeError(
        "Too few valid calibration points. "
        "Improve lighting and camera position."
    )

expected_features = len(
    feature_names(USE_HEAD_POSE)
)

if features_all.shape[1] != expected_features:
    raise RuntimeError(
        f"Feature mismatch: received "
        f"{features_all.shape[1]}, expected "
        f"{expected_features}."
    )


models = {
    "ridge_linear": make_pipeline(
        StandardScaler(),
        Ridge(alpha=2.0),
    ),

    "ridge_polynomial": make_pipeline(
        PolynomialFeatures(
            degree=2,
            include_bias=False,
        ),
        StandardScaler(),
        Ridge(alpha=5.0),
    ),

    "svr_rbf": make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(
            SVR(
                kernel="rbf",
                C=30.0,
                epsilon=8.0,
            )
        ),
    ),
}


print()
print("==============================================")
print("MODEL EVALUATION")
print("==============================================")

scores = {}

for name, candidate in models.items():
    result = evaluate_model(
        candidate,
        features_all,
        targets_all,
        groups_all,
    )

    scores[name] = result

    print(
        f"{name}: "
        f"mean={result['mean']:.1f}px, "
        f"median={result['median']:.1f}px, "
        f"worst={result['worst']:.1f}px"
    )


best_name = min(
    scores,
    key=lambda name: scores[name]["mean"],
)

final_model = models[best_name]

final_model.fit(
    features_all,
    targets_all,
)

joblib.dump(
    final_model,
    "gaze_model.pkl",
)


with open(
    "calibration_raw.csv",
    "w",
    newline="",
) as file:
    writer = csv.writer(file)

    writer.writerow(
        feature_names(USE_HEAD_POSE)
        + [
            "target_x",
            "target_y",
            "point_id",
        ]
    )

    for values, target, group in zip(
        features_all,
        targets_all,
        groups_all,
    ):
        writer.writerow(
            list(values)
            + [
                int(target[0]),
                int(target[1]),
                int(group),
            ]
        )


metadata = {
    "feature_version": FEATURE_VERSION,
    "feature_names": feature_names(USE_HEAD_POSE),
    "feature_count": int(
        features_all.shape[1]
    ),
    "use_head_pose": USE_HEAD_POSE,
    "model_type": best_name,
    "cv_mean_error_px": scores[best_name]["mean"],
    "cv_median_error_px": scores[best_name]["median"],
    "cv_worst_error_px": scores[best_name]["worst"],
    "cv_all_models_px": scores,
    "screen_width": SCREEN_WIDTH,
    "screen_height": SCREEN_HEIGHT,
    "n_calibration_points": valid_points,
    "n_samples": int(len(features_all)),
}

with open(
    "gaze_model_metadata.json",
    "w",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


print()
print("==============================================")
print("CALIBRATION COMPLETE")
print("==============================================")
print("Best model:", best_name)
print(
    f"Mean error: "
    f"{scores[best_name]['mean']:.1f}px"
)
print(
    f"Median error: "
    f"{scores[best_name]['median']:.1f}px"
)
print(
    f"Worst error: "
    f"{scores[best_name]['worst']:.1f}px"
)
print("Saved gaze_model.pkl")
print("Saved gaze_model_metadata.json")