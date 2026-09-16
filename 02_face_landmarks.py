import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# 1. Location of the MediaPipe model
# --------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"


# --------------------------------------------------
# 2. Configure MediaPipe Face Landmarker
# --------------------------------------------------

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
# 3. Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    detector.close()
    exit()

print("Face landmark tracking started.")
print("Press Q to quit.")


# MediaPipe video timestamps must increase
timestamp_ms = 0


# --------------------------------------------------
# 4. Process webcam frames
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)

    # OpenCV gives BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face landmarks
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    timestamp_ms += 33


    # --------------------------------------------------
    # 5. Draw the landmarks
    # --------------------------------------------------

    if result.face_landmarks:

        height, width, _ = frame.shape

        for face_landmarks in result.face_landmarks:

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )


    # Display
    cv2.imshow(
        "Face Landmarks",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 6. Clean up
# --------------------------------------------------

cap.release()
detector.close()
cv2.destroyAllWindows()