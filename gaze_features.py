import numpy as np


FEATURE_VERSION = "v5_eye_features_4"


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

MIN_EYE_WIDTH = 0.001


def feature_names(use_head_pose=False):
    names = [
        "left_h",
        "left_v",
        "right_h",
        "right_v",
    ]

    if use_head_pose:
        names += [
            "yaw",
            "pitch",
            "roll",
        ]

    return names


def _iris_center(landmarks, indices):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]

    return float(np.mean(xs)), float(np.mean(ys))


def _eye_normalized(landmarks, corner_ids, iris_xy):
    c1 = landmarks[corner_ids[0]]
    c2 = landmarks[corner_ids[1]]

    center_x = (c1.x + c2.x) / 2.0
    center_y = (c1.y + c2.y) / 2.0

    width = abs(c2.x - c1.x)

    if width < MIN_EYE_WIDTH:
        return None

    iris_x, iris_y = iris_xy

    horizontal = (iris_x - center_x) / width
    vertical = (iris_y - center_y) / width

    return float(horizontal), float(vertical)


def get_features(face_landmarks, facial_transformation_matrix=None):
    left_iris = _iris_center(face_landmarks, LEFT_IRIS)
    right_iris = _iris_center(face_landmarks, RIGHT_IRIS)

    left = _eye_normalized(
        face_landmarks,
        LEFT_EYE_CORNERS,
        left_iris,
    )

    right = _eye_normalized(
        face_landmarks,
        RIGHT_EYE_CORNERS,
        right_iris,
    )

    if left is None or right is None:
        return None

    features = np.asarray(
        [
            left[0],
            left[1],
            right[0],
            right[1],
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(features)):
        return None

    return features.tolist()