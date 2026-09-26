import numpy as np


FEATURE_VERSION = "v5_eye_features_4"


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

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


def _eye_geometry(landmarks, corner_ids):
    c1 = landmarks[corner_ids[0]]
    c2 = landmarks[corner_ids[1]]

    center_x = (c1.x + c2.x) / 2.0
    center_y = (c1.y + c2.y) / 2.0
    width = abs(c2.x - c1.x)
    height = abs(c2.y - c1.y)

    if width < MIN_EYE_WIDTH:
        return None

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width": float(width),
        "height": float(height),
        "left": float(min(c1.x, c2.x)),
        "right": float(max(c1.x, c2.x)),
        "top": float(min(c1.y, c2.y)),
        "bottom": float(max(c1.y, c2.y)),
    }


def _eye_normalized(landmarks, corner_ids, iris_xy):
    eye_geometry = _eye_geometry(landmarks, corner_ids)
    if eye_geometry is None:
        return None

    iris_x, iris_y = iris_xy
    horizontal = (iris_x - eye_geometry["center_x"]) / eye_geometry["width"]
    vertical = (iris_y - eye_geometry["center_y"]) / eye_geometry["width"]
    return float(horizontal), float(vertical)


def _eye_aspect_ratio(landmarks, eye_top_index, eye_bottom_index, corner_ids):
    top = landmarks[eye_top_index]
    bottom = landmarks[eye_bottom_index]
    c1 = landmarks[corner_ids[0]]
    c2 = landmarks[corner_ids[1]]

    vertical = np.hypot(top.x - bottom.x, top.y - bottom.y)
    horizontal = abs(c2.x - c1.x)
    if horizontal < MIN_EYE_WIDTH:
        return 0.0
    return float(vertical / horizontal)


def head_pose_from_matrix(matrix4x4):
    if matrix4x4 is None:
        return 0.0, 0.0, 0.0

    try:
        matrix = np.asarray(matrix4x4, dtype=float)
    except Exception:
        return 0.0, 0.0, 0.0

    if matrix.size == 0 or matrix.shape not in {(3, 3), (4, 4)}:
        return 0.0, 0.0, 0.0

    rotation = matrix[:3, :3] if matrix.shape == (4, 4) else matrix
    if rotation.shape != (3, 3):
        return 0.0, 0.0, 0.0

    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    return yaw, pitch, roll


def extract_eye_metrics(face_landmarks, facial_transformation_matrix=None):
    left_iris = _iris_center(face_landmarks, LEFT_IRIS)
    right_iris = _iris_center(face_landmarks, RIGHT_IRIS)

    left = _eye_normalized(face_landmarks, LEFT_EYE_CORNERS, left_iris)
    right = _eye_normalized(face_landmarks, RIGHT_EYE_CORNERS, right_iris)

    if left is None or right is None:
        return None

    left_geometry = _eye_geometry(face_landmarks, LEFT_EYE_CORNERS)
    right_geometry = _eye_geometry(face_landmarks, RIGHT_EYE_CORNERS)
    if left_geometry is None or right_geometry is None:
        return None

    left_ear = _eye_aspect_ratio(face_landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_CORNERS)
    right_ear = _eye_aspect_ratio(face_landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_CORNERS)
    eye_aspect_ratio = float((left_ear + right_ear) / 2.0)

    yaw, pitch, roll = head_pose_from_matrix(facial_transformation_matrix)

    normalized_iris_x = float((left[0] + right[0]) / 2.0)
    normalized_iris_y = float((left[1] + right[1]) / 2.0)

    metrics = {
        "left_iris_center": left_iris,
        "right_iris_center": right_iris,
        "left_eye_geometry": left_geometry,
        "right_eye_geometry": right_geometry,
        "left_horizontal": float(left[0]),
        "left_vertical": float(left[1]),
        "right_horizontal": float(right[0]),
        "right_vertical": float(right[1]),
        "normalized_iris_x": normalized_iris_x,
        "normalized_iris_y": normalized_iris_y,
        "eye_aspect_ratio": eye_aspect_ratio,
        "eye_openness": eye_aspect_ratio,
        "blink_state": "OPEN" if eye_aspect_ratio > 0.15 else "CLOSED",
        "head_yaw": float(yaw),
        "head_pitch": float(pitch),
        "head_roll": float(roll),
    }

    if not all(np.isfinite(value) for value in np.asarray(list(metrics["left_eye_geometry"].values()) + list(metrics["right_eye_geometry"].values()) + [
        metrics["left_horizontal"],
        metrics["left_vertical"],
        metrics["right_horizontal"],
        metrics["right_vertical"],
        metrics["normalized_iris_x"],
        metrics["normalized_iris_y"],
        metrics["eye_aspect_ratio"],
        metrics["head_yaw"],
        metrics["head_pitch"],
        metrics["head_roll"],
    ], dtype=float)):
        return None

    return metrics


def get_features(face_landmarks, facial_transformation_matrix=None):
    metrics = extract_eye_metrics(face_landmarks, facial_transformation_matrix)
    if metrics is None:
        return None

    features = [
        metrics["left_horizontal"],
        metrics["left_vertical"],
        metrics["right_horizontal"],
        metrics["right_vertical"],
    ]

    if facial_transformation_matrix is not None:
        features += [
            metrics["head_yaw"],
            metrics["head_pitch"],
            metrics["head_roll"],
        ]

    return [float(value) for value in features]


def estimate_gaze_confidence(metrics):
    if not isinstance(metrics, dict):
        return 0.0

    left_geometry = metrics.get("left_eye_geometry", {})
    right_geometry = metrics.get("right_eye_geometry", {})

    width_score = 1.0
    if left_geometry and right_geometry:
        left_width = float(left_geometry.get("width", 0.0))
        right_width = float(right_geometry.get("width", 0.0))
        avg_width = (left_width + right_width) / 2.0
        width_score = float(np.clip(avg_width / 0.18, 0.0, 1.0))

    symmetry_penalty = 0.0
    left_h = float(metrics.get("left_horizontal", 0.0))
    right_h = float(metrics.get("right_horizontal", 0.0))
    left_v = float(metrics.get("left_vertical", 0.0))
    right_v = float(metrics.get("right_vertical", 0.0))
    symmetry_penalty += abs(left_h - right_h)
    symmetry_penalty += abs(left_v - right_v)
    symmetry_score = float(np.clip(1.0 - (symmetry_penalty / 0.8), 0.0, 1.0))

    yaw = abs(float(metrics.get("head_yaw", 0.0)))
    pitch = abs(float(metrics.get("head_pitch", 0.0)))
    roll = abs(float(metrics.get("head_roll", 0.0)))
    head_pose_score = float(np.clip(1.0 - (yaw + pitch + roll) / 1.5, 0.0, 1.0))

    eye_open = float(metrics.get("eye_aspect_ratio", 0.0))
    openness_score = float(np.clip((eye_open - 0.05) / 0.35, 0.0, 1.0))

    score = (
        0.35 * width_score
        + 0.30 * symmetry_score
        + 0.20 * head_pose_score
        + 0.15 * openness_score
    )

    return float(np.clip(score, 0.0, 1.0))