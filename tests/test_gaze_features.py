import numpy as np

from gaze_features import (
    estimate_gaze_confidence,
    extract_eye_metrics,
    feature_names,
    get_features,
)


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_extract_eye_metrics_returns_quality_metrics():
    face = [
        Landmark(0.50, 0.50),
    ] * 478

    left_eye = [33, 133]
    right_eye = [362, 263]
    left_iris = [468, 469, 470, 471, 472]
    right_iris = [473, 474, 475, 476, 477]

    for idx in left_eye:
        face[idx] = Landmark(0.40, 0.45)
    face[33] = Landmark(0.35, 0.45)
    face[133] = Landmark(0.45, 0.45)

    for idx in right_eye:
        face[idx] = Landmark(0.65, 0.45)
    face[362] = Landmark(0.60, 0.45)
    face[263] = Landmark(0.70, 0.45)

    for idx in left_iris:
        face[idx] = Landmark(0.38 + (idx - 468) * 0.01, 0.47)
    for idx in right_iris:
        face[idx] = Landmark(0.62 + (idx - 473) * 0.01, 0.47)

    metrics = extract_eye_metrics(face)

    assert metrics is not None
    assert set(metrics.keys()) >= {
        "left_iris_center",
        "right_iris_center",
        "normalized_iris_x",
        "normalized_iris_y",
        "left_eye_geometry",
        "right_eye_geometry",
        "eye_aspect_ratio",
        "head_yaw",
        "head_pitch",
        "head_roll",
    }
    assert np.isfinite(metrics["normalized_iris_x"]) and np.isfinite(metrics["normalized_iris_y"])


def test_get_features_supports_head_pose_and_compatible_output():
    face = [Landmark(0.5, 0.5) for _ in range(478)]
    face[33] = Landmark(0.40, 0.48)
    face[133] = Landmark(0.50, 0.48)
    face[362] = Landmark(0.60, 0.48)
    face[263] = Landmark(0.70, 0.48)

    for idx in range(468, 473):
        face[idx] = Landmark(0.42 + (idx - 468) * 0.02, 0.47)
    for idx in range(473, 478):
        face[idx] = Landmark(0.58 + (idx - 473) * 0.02, 0.47)

    head_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)

    vector = get_features(face, head_pose)
    assert vector is not None
    assert len(vector) == 7
    assert feature_names(use_head_pose=True) == [
        "left_h",
        "left_v",
        "right_h",
        "right_v",
        "yaw",
        "pitch",
        "roll",
    ]
    assert np.all(np.isfinite(vector))


def test_estimate_gaze_confidence_returns_bounded_score_for_valid_metrics():
    metrics = {
        "left_eye_geometry": {"width": 0.18, "height": 0.08},
        "right_eye_geometry": {"width": 0.18, "height": 0.08},
        "left_horizontal": 0.05,
        "left_vertical": 0.04,
        "right_horizontal": -0.06,
        "right_vertical": 0.03,
        "head_yaw": 0.12,
        "head_pitch": 0.10,
        "head_roll": 0.04,
        "eye_aspect_ratio": 0.30,
    }

    score = estimate_gaze_confidence(metrics)
    assert 0.0 <= score <= 1.0
    assert score > 0.5
