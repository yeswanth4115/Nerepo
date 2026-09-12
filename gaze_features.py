import numpy as np

FEATURE_VERSION = "v3_eye_center"

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]


def feature_names(use_head_pose=False):
    names = ["left_center_x", "left_center_y", "right_center_x", "right_center_y"]
    if use_head_pose:
        names.extend(
            ["pose_tx", "pose_ty", "pose_tz", "pose_rx", "pose_ry", "pose_rz"]
        )
    return names


def get_features(face_landmarks, facial_matrix=None):
    left_iris_x = np.mean([face_landmarks[i].x for i in LEFT_IRIS])
    left_iris_y = np.mean([face_landmarks[i].y for i in LEFT_IRIS])
    right_iris_x = np.mean([face_landmarks[i].x for i in RIGHT_IRIS])
    right_iris_y = np.mean([face_landmarks[i].y for i in RIGHT_IRIS])

    # Normalize iris position relative to the eye center and width. Using
    # width for both axes avoids eyelid opening changing the vertical signal.
    c1_l = face_landmarks[LEFT_EYE_CORNERS[0]]
    c2_l = face_landmarks[LEFT_EYE_CORNERS[1]]
    left_center_x = (c1_l.x + c2_l.x) / 2
    left_center_y = (c1_l.y + c2_l.y) / 2
    left_width = abs(c2_l.x - c1_l.x)
    if left_width < 0.001:
        return None
    left_norm_x = (left_iris_x - left_center_x) / left_width
    left_norm_y = (left_iris_y - left_center_y) / left_width

    # Right eye normalization
    c1_r = face_landmarks[RIGHT_EYE_CORNERS[0]]
    c2_r = face_landmarks[RIGHT_EYE_CORNERS[1]]
    right_center_x = (c1_r.x + c2_r.x) / 2
    right_center_y = (c1_r.y + c2_r.y) / 2
    right_width = abs(c2_r.x - c1_r.x)
    if right_width < 0.001:
        return None
    right_norm_x = (right_iris_x - right_center_x) / right_width
    right_norm_y = (right_iris_y - right_center_y) / right_width

    feats = [left_norm_x, left_norm_y, right_norm_x, right_norm_y]

    if facial_matrix is not None:
        mat = np.array(facial_matrix)
        tx, ty, tz = mat[0, 3], mat[1, 3], mat[2, 3]
        rx = np.arctan2(mat[2, 1], mat[2, 2])
        ry = np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
        rz = np.arctan2(mat[1, 0], mat[0, 0])
        feats.extend([tx, ty, tz, rx, ry, rz])

    return feats