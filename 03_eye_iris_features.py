import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. MediaPipe model
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Iris landmark indices
# --------------------------------------------------

# MediaPipe Face Landmarker has 478 landmarks when
# iris refinement is included.

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


# --------------------------------------------------
# 3. Calculate center of iris
# --------------------------------------------------

def iris_center(face_landmarks, indices, width, height):

    x_sum = 0
    y_sum = 0

    for index in indices:

        landmark = face_landmarks[index]

        x_sum += landmark.x
        y_sum += landmark.y

    x = x_sum / len(indices)
    y = y_sum / len(indices)

    pixel_x = int(x * width)
    pixel_y = int(y * height)

    return pixel_x, pixel_y, x, y


# --------------------------------------------------
# 4. Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    detector.close()
    exit()

print("Eye and iris tracking started.")
print("Press Q to quit.")

timestamp_ms = 0


# --------------------------------------------------
# 5. Process frames
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect landmarks
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    timestamp_ms += 33


    # --------------------------------------------------
    # 6. Extract iris coordinates
    # --------------------------------------------------

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        # Left iris
        left_x, left_y, left_nx, left_ny = iris_center(
            face_landmarks,
            LEFT_IRIS,
            width,
            height
        )

        # Right iris
        right_x, right_y, right_nx, right_ny = iris_center(
            face_landmarks,
            RIGHT_IRIS,
            width,
            height
        )


        # --------------------------------------------------
        # 7. Draw iris centers
        # --------------------------------------------------

        cv2.circle(
            frame,
            (left_x, left_y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (right_x, right_y),
            6,
            (0, 0, 255),
            -1
        )


        # --------------------------------------------------
        # 8. Display coordinates
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Left Iris: ({left_x}, {left_y})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Right Iris: ({right_x}, {right_y})",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # Print values to terminal
        print(
            f"Left=({left_x},{left_y}) "
            f"Right=({right_x},{right_y})"
        )


    # Display
    cv2.imshow(
        "Eye and Iris Tracking",
        frame
    )


    # Q → quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 9. Clean up
# --------------------------------------------------

cap.release()
detector.close()
cv2.destroyAllWindows()