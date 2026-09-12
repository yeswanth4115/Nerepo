import numpy as np

FEATURE_VERSION = "center_v2_headpose"

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

MIN_EYE_WIDTH = 0.001


def _iris_center(landmarks, indices):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def _eye_normalized(landmarks, corner_ids, iris_xy):
    """Center-relative normalization: (iris - eye_center) / eye_width.

    Chosen over box-relative ((iris - min_corner) / width) because it's
    less sensitive to which corner MediaPipe calls "min" vs "max" as the
    head turns slightly, and it's symmetric around 0 which plays nicer
    with regularized linear models.
    """
    c1 = landmarks[corner_ids[0]]
    c2 = landmarks[corner_ids[1]]

    center_x = (c1.x + c2.x) / 2
    center_y = (c1.y + c2.y) / 2
    width = abs(c2.x - c1.x)

    if width < MIN_EYE_WIDTH:
        return None

    iris_x, iris_y = iris_xy
    h = (iris_x - center_x) / width
    v = (iris_y - center_y) / width
    return h, v


def head_pose_from_matrix(matrix4x4):
    """Approximate yaw/pitch/roll (radians) from MediaPipe's
    facial_transformation_matrix.

    This is a *relative* feature to help the regressor compensate for
    head movement between calibration and use — not a metrology-grade
    pose estimate. Don't use these numbers for anything that needs
    real angular accuracy.
    """
    R = np.array(matrix4x4)[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


def get_features(face_landmarks, facial_transformation_matrix=None):
    """
    Returns a 4-element [left_h, left_v, right_h, right_v] feature
    vector, extended to 7 elements with [yaw, pitch, roll] if a
    facial_transformation_matrix is provided. Returns None if
    extraction failed (occluded eye, corner collapse, etc).
    """
    left_iris = _iris_center(face_landmarks, LEFT_IRIS)
    right_iris = _iris_center(face_landmarks, RIGHT_IRIS)

    left = _eye_normalized(face_landmarks, LEFT_EYE_CORNERS, left_iris)
    right = _eye_normalized(face_landmarks, RIGHT_EYE_CORNERS, right_iris)

    if left is None or right is None:
        return None

    features = list(left) + list(right)

    if facial_transformation_matrix is not None:
        yaw, pitch, roll = head_pose_from_matrix(facial_transformation_matrix)
        features += [yaw, pitch, roll]

    return features


def feature_names(use_head_pose):
    names = ["left_h", "left_v", "right_h", "right_v"]
    if use_head_pose:
        names += ["yaw", "pitch", "roll"]
    return names
