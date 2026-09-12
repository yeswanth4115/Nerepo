import cv2
import mediapipe as mp
import numpy as np
import csv
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# 1. MEDIAPIPE MODEL
# ==========================================================

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
    min_tracking_confidence=0.5
)

detector = vision.FaceLandmarker.create_from_options(options)


# ==========================================================
# 2. LANDMARKS
# ==========================================================

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corners
LEFT_CORNER_1 = 33
LEFT_CORNER_2 = 133

RIGHT_CORNER_1 = 362
RIGHT_CORNER_2 = 263


# ==========================================================
# 3. IMPROVED FEATURE EXTRACTION
# ==========================================================

def get_features(face_landmarks):

    # ------------------------------------------------------
    # LEFT IRIS CENTER
    # ------------------------------------------------------

    left_iris_x = np.mean([
        face_landmarks[i].x
        for i in LEFT_IRIS
    ])

    left_iris_y = np.mean([
        face_landmarks[i].y
        for i in LEFT_IRIS
    ])


    # ------------------------------------------------------
    # RIGHT IRIS CENTER
    # ------------------------------------------------------

    right_iris_x = np.mean([
        face_landmarks[i].x
        for i in RIGHT_IRIS
    ])

    right_iris_y = np.mean([
        face_landmarks[i].y
        for i in RIGHT_IRIS
    ])


    # ======================================================
    # LEFT EYE
    # ======================================================

    left_corner_1 = face_landmarks[LEFT_CORNER_1]
    left_corner_2 = face_landmarks[LEFT_CORNER_2]


    # Eye center

    left_eye_center_x = (
        left_corner_1.x +
        left_corner_2.x
    ) / 2

    left_eye_center_y = (
        left_corner_1.y +
        left_corner_2.y
    ) / 2


    # Eye width

    left_eye_width = abs(
        left_corner_2.x -
        left_corner_1.x
    )


    if left_eye_width < 0.001:
        return None


    # Normalize using eye width

    left_horizontal = (
        left_iris_x -
        left_eye_center_x
    ) / left_eye_width


    left_vertical = (
        left_iris_y -
        left_eye_center_y
    ) / left_eye_width


    # ======================================================
    # RIGHT EYE
    # ======================================================

    right_corner_1 = face_landmarks[
        RIGHT_CORNER_1
    ]

    right_corner_2 = face_landmarks[
        RIGHT_CORNER_2
    ]


    # Eye center

    right_eye_center_x = (
        right_corner_1.x +
        right_corner_2.x
    ) / 2

    right_eye_center_y = (
        right_corner_1.y +
        right_corner_2.y
    ) / 2


    # Eye width

    right_eye_width = abs(
        right_corner_2.x -
        right_corner_1.x
    )


    if right_eye_width < 0.001:
        return None


    # Normalize using eye width

    right_horizontal = (
        right_iris_x -
        right_eye_center_x
    ) / right_eye_width


    right_vertical = (
        right_iris_y -
        right_eye_center_y
    ) / right_eye_width


    return (
        left_horizontal,
        left_vertical,
        right_horizontal,
        right_vertical
    )


# ==========================================================
# 4. CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    exit()


# ==========================================================
# 5. CSV
# ==========================================================

filename = "improved_eye_features.csv"

csv_file = open(
    filename,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Direction",
    "Left_Horizontal",
    "Left_Vertical",
    "Right_Horizontal",
    "Right_Vertical"
])


# ==========================================================
# 6. DIRECTIONS
# ==========================================================

directions = [
    "CENTER",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT"
]


# ==========================================================
# 7. WINDOW
# ==========================================================

WINDOW_NAME = "Improved Eye Feature Collection"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

timestamp_ms = 0

total_samples = 0


# ==========================================================
# 8. START
# ==========================================================

print()
print("============================================")
print("IMPROVED EYE FEATURE COLLECTION")
print("============================================")
print()
print("Keep your HEAD completely still.")
print("Move ONLY your eyes.")
print()
print("Starting in 3 seconds...")
print()

time.sleep(3)


# ==========================================================
# 9. COLLECT DATA
# ==========================================================

for direction in directions:

    print()
    print("--------------------------------------------")
    print("LOOK:", direction)
    print("--------------------------------------------")


    # ------------------------------------------------------
    # Instruction period
    # ------------------------------------------------------

    instruction_start = time.time()

    while time.time() - instruction_start < 2:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            f"LOOK {direction}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 255),
            4
        )

        cv2.putText(
            frame,
            "Keep your head still",
            (50, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            csv_file.close()
            cap.release()
            detector.close()
            cv2.destroyAllWindows()

            exit()


    # ------------------------------------------------------
    # Collect for 4 seconds
    # ------------------------------------------------------

    collection_start = time.time()

    direction_samples = 0


    while time.time() - collection_start < 4:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        timestamp_ms += 33


        if result.face_landmarks:

            features = get_features(
                result.face_landmarks[0]
            )


            if features is not None:

                lh, lv, rh, rv = features


                # Save

                writer.writerow([
                    direction,
                    lh,
                    lv,
                    rh,
                    rv
                ])

                csv_file.flush()


                direction_samples += 1
                total_samples += 1


                # --------------------------------------------------
                # Display
                # --------------------------------------------------

                cv2.putText(
                    frame,
                    f"Direction: {direction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Left H:{lh:.3f} V:{lv:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right H:{rh:.3f} V:{rv:.3f}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Samples: {direction_samples}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )


        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            csv_file.close()
            cap.release()
            detector.close()
            cv2.destroyAllWindows()

            exit()


    print(
        f"{direction}: {direction_samples} samples"
    )


# ==========================================================
# 10. CLEANUP
# ==========================================================

csv_file.close()

cap.release()

detector.close()

cv2.destroyAllWindows()


print()
print("============================================")
print("COLLECTION COMPLETE")
print("============================================")
print()
print("Total samples:", total_samples)
print()
print("Saved:")
print("improved_eye_features.csv")
print()