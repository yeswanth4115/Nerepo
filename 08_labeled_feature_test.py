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
# 2. LANDMARK INDICES
# ==========================================================

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_VERTICAL = [159, 145]
RIGHT_EYE_VERTICAL = [386, 374]


# ==========================================================
# 3. FEATURE EXTRACTION
# ==========================================================

def get_features(face_landmarks):

    left_iris_x = np.mean(
        [face_landmarks[i].x for i in LEFT_IRIS]
    )

    left_iris_y = np.mean(
        [face_landmarks[i].y for i in LEFT_IRIS]
    )

    right_iris_x = np.mean(
        [face_landmarks[i].x for i in RIGHT_IRIS]
    )

    right_iris_y = np.mean(
        [face_landmarks[i].y for i in RIGHT_IRIS]
    )

    # ---------- LEFT EYE ----------

    left_min_x = min(
        face_landmarks[33].x,
        face_landmarks[133].x
    )

    left_max_x = max(
        face_landmarks[33].x,
        face_landmarks[133].x
    )

    left_width = left_max_x - left_min_x

    if left_width < 0.001:
        return None

    left_x = (
        left_iris_x - left_min_x
    ) / left_width


    left_top = min(
        face_landmarks[159].y,
        face_landmarks[145].y
    )

    left_bottom = max(
        face_landmarks[159].y,
        face_landmarks[145].y
    )

    left_height = left_bottom - left_top

    if left_height < 0.001:
        return None

    left_y = (
        left_iris_y - left_top
    ) / left_height


    # ---------- RIGHT EYE ----------

    right_min_x = min(
        face_landmarks[362].x,
        face_landmarks[263].x
    )

    right_max_x = max(
        face_landmarks[362].x,
        face_landmarks[263].x
    )

    right_width = right_max_x - right_min_x

    if right_width < 0.001:
        return None

    right_x = (
        right_iris_x - right_min_x
    ) / right_width


    right_top = min(
        face_landmarks[386].y,
        face_landmarks[374].y
    )

    right_bottom = max(
        face_landmarks[386].y,
        face_landmarks[374].y
    )

    right_height = right_bottom - right_top

    if right_height < 0.001:
        return None

    right_y = (
        right_iris_y - right_top
    ) / right_height


    return left_x, left_y, right_x, right_y


# ==========================================================
# 4. CAMERA
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    exit()


# ==========================================================
# 5. DATA FILE
# ==========================================================

filename = "labeled_eye_features.csv"

csv_file = open(
    filename,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Direction",
    "Left_X",
    "Left_Y",
    "Right_X",
    "Right_Y"
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

WINDOW_NAME = "Labeled Eye Feature Collection"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


timestamp_ms = 0

total_samples = 0


print()
print("============================================")
print("LABELED EYE FEATURE COLLECTION")
print("============================================")
print()
print("Keep your HEAD completely still.")
print("Move ONLY your eyes.")
print()
print("The program will tell you where to look.")
print()
print("Starting in 3 seconds...")
print()

time.sleep(3)


# ==========================================================
# 8. COLLECT EACH DIRECTION
# ==========================================================

for direction in directions:

    print()
    print("--------------------------------------------")
    print("LOOK:", direction)
    print("--------------------------------------------")

    # ------------------------------------------
    # Instruction period
    # ------------------------------------------

    instruction_start = time.time()

    while time.time() - instruction_start < 2:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Darken image slightly

        frame = cv2.convertScaleAbs(
            frame,
            alpha=0.7,
            beta=0
        )

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


    # ------------------------------------------
    # Collect data for 4 seconds
    # ------------------------------------------

    print("Collecting samples...")

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

                lx, ly, rx, ry = features

                writer.writerow([
                    direction,
                    lx,
                    ly,
                    rx,
                    ry
                ])

                csv_file.flush()

                direction_samples += 1
                total_samples += 1


                # Display features

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
                    f"Left  X:{lx:.3f} Y:{ly:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right X:{rx:.3f} Y:{ry:.3f}",
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
        f"{direction}: {direction_samples} samples collected"
    )


# ==========================================================
# 9. CLEANUP
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
print("Saved file:")
print("labeled_eye_features.csv")
print()