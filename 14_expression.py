import cv2
import mediapipe as mp
import time
import random
import os
import numpy as np
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"
ROUND_DURATION = 5
STABLE_FRAMES = 6
CALIBRATION_SECONDS = 3

EXPRESSIONS = ["HAPPY", "SURPRISED", "NEUTRAL"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPRESSION_DIR = os.path.join(BASE_DIR, "expression")

expression_images = {
    "HAPPY": cv2.imread(
        os.path.join(EXPRESSION_DIR, "happy.png"),
        cv2.IMREAD_UNCHANGED,
    ),
    "SURPRISED": cv2.imread(
        os.path.join(EXPRESSION_DIR, "surprised.png"),
        cv2.IMREAD_UNCHANGED,
    ),
    "NEUTRAL": cv2.imread(
        os.path.join(EXPRESSION_DIR, "neutral.png"),
        cv2.IMREAD_UNCHANGED,
    ),
}

for name, image in expression_images.items():
    if image is None:
        raise RuntimeError(
            f"Could not load {name}.png from {EXPRESSION_DIR}"
        )


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    output_face_blendshapes=True,
)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    detector.close()
    raise RuntimeError("Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WINDOW_NAME = "Expression Imitation Game"
timestamp_ms = 0


def next_timestamp():
    global timestamp_ms
    timestamp_ms += 33
    return timestamp_ms


def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def display_png(frame, image, center_x, center_y):
    height, width = image.shape[:2]

    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = x1 + width
    y2 = y1 + height

    frame_height, frame_width = frame.shape[:2]

    source_x1 = max(0, -x1)
    source_y1 = max(0, -y1)
    source_x2 = width - max(0, x2 - frame_width)
    source_y2 = height - max(0, y2 - frame_height)

    destination_x1 = max(0, x1)
    destination_y1 = max(0, y1)
    destination_x2 = min(frame_width, x2)
    destination_y2 = min(frame_height, y2)

    if destination_x1 >= destination_x2:
        return

    if destination_y1 >= destination_y2:
        return

    image_crop = image[
        source_y1:source_y2,
        source_x1:source_x2,
    ]

    background = frame[
        destination_y1:destination_y2,
        destination_x1:destination_x2,
    ]

    if image_crop.shape[2] == 4:
        alpha = image_crop[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]

        foreground = image_crop[:, :, :3].astype(np.float32)
        background_float = background.astype(np.float32)

        blended = (
            foreground * alpha
            + background_float * (1.0 - alpha)
        )

        frame[
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = blended.astype(np.uint8)
    else:
        frame[
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = image_crop


def get_blendshape_values(result):
    if not result.face_blendshapes:
        return {}

    return {
        category.category_name: category.score
        for category in result.face_blendshapes[0]
    }


def get_expression_features(blendshapes):
    return {
        "smile": (
            blendshapes.get("mouthSmileLeft", 0.0)
            + blendshapes.get("mouthSmileRight", 0.0)
        ) / 2.0,

        "cheek": (
            blendshapes.get("cheekSquintLeft", 0.0)
            + blendshapes.get("cheekSquintRight", 0.0)
        ) / 2.0,

        "jaw": blendshapes.get("jawOpen", 0.0),

        "brow": (
            blendshapes.get("browInnerUp", 0.0)
            + blendshapes.get("browOuterUpLeft", 0.0)
            + blendshapes.get("browOuterUpRight", 0.0)
        ) / 3.0,
    }


def average_features(feature_samples):
    if not feature_samples:
        return None

    return {
        key: float(np.mean([
            sample[key]
            for sample in feature_samples
        ]))
        for key in feature_samples[0]
    }


def calibrate_neutral():
    print("Calibration started.")
    print("Keep your face relaxed and look at the camera.")

    samples = []
    start_time = time.time()

    while time.time() - start_time < CALIBRATION_SECONDS:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = detector.detect_for_video(
            mp_image,
            next_timestamp(),
        )

        if result.face_landmarks:
            blendshapes = get_blendshape_values(result)

            if blendshapes:
                samples.append(
                    get_expression_features(blendshapes)
                )

        cv2.putText(
            frame,
            "Keep your face neutral...",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            detector.close()
            cv2.destroyAllWindows()
            raise SystemExit

    if len(samples) < 10:
        raise RuntimeError(
            "Calibration failed. Make sure your face is visible."
        )

    baseline = {}

    for key in samples[0]:
        baseline[key] = float(np.median([
            sample[key]
            for sample in samples
        ]))

    print("Calibration completed.")
    print("Neutral baseline:", baseline)

    return baseline


def detect_expression(blendshapes, baseline, feature_history):
    if not blendshapes:
        return "NO FACE", 0.0

    current_features = get_expression_features(blendshapes)
    feature_history.append(current_features)

    averaged = average_features(feature_history)

    smile_delta = max(
        0.0,
        averaged["smile"] - baseline["smile"],
    )

    cheek_delta = max(
        0.0,
        averaged["cheek"] - baseline["cheek"],
    )

    jaw_delta = max(
        0.0,
        averaged["jaw"] - baseline["jaw"],
    )

    brow_delta = max(
        0.0,
        averaged["brow"] - baseline["brow"],
    )

    happy_score = min(
        1.0,
        (smile_delta / 0.16) * 0.7
        + (cheek_delta / 0.12) * 0.3,
    )

    surprised_score = min(
        1.0,
        (jaw_delta / 0.18) * 0.65
        + (brow_delta / 0.14) * 0.35,
    )

    activity = max(
        smile_delta / 0.16,
        cheek_delta / 0.12,
        jaw_delta / 0.18,
        brow_delta / 0.14,
    )

    neutral_score = max(
        0.0,
        1.0 - min(1.0, activity),
    )

    scores = {
        "HAPPY": happy_score,
        "SURPRISED": surprised_score,
        "NEUTRAL": neutral_score,
    }

    detected = max(scores, key=scores.get)

    sorted_scores = sorted(
        scores.values(),
        reverse=True,
    )

    confidence = sorted_scores[0] - sorted_scores[1]

    if detected == "NEUTRAL":
        confidence = neutral_score

    confidence = float(np.clip(confidence, 0.0, 1.0))

    return detected, confidence


def draw_text(frame, text, position, size=0.8, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        2,
        cv2.LINE_AA,
    )


print("Expression imitation game starting.")
print("Press Q to quit.")

neutral_baseline = calibrate_neutral()

current_expression = random.choice(EXPRESSIONS)
score = 0
round_number = 1
round_start_time = time.time()
round_correct = False

last_expression = "NO FACE"
expression_streak = 0
feature_history = deque(maxlen=5)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = detector.detect_for_video(
            mp_image,
            next_timestamp(),
        )

        detected_expression = "NO FACE"
        detection_confidence = 0.0

        if result.face_landmarks:
            draw_text(
                frame,
                "FACE DETECTED",
                (20, 40),
                color=(0, 255, 0),
            )

            blendshapes = get_blendshape_values(result)

            detected_expression, detection_confidence = (
                detect_expression(
                    blendshapes,
                    neutral_baseline,
                    feature_history,
                )
            )
        else:
            feature_history.clear()

            draw_text(
                frame,
                "SHOW YOUR FACE",
                (20, 40),
                color=(0, 0, 255),
            )

        if detected_expression == last_expression:
            expression_streak += 1
        else:
            expression_streak = 1
            last_expression = detected_expression

        stable_expression = detected_expression

        if expression_streak < STABLE_FRAMES:
            stable_expression = "HOLD"

        is_correct_now = (
            stable_expression == current_expression
            and detection_confidence >= 0.20
        )

        if is_correct_now:
            round_correct = True

        draw_text(
            frame,
            f"Round: {round_number}",
            (20, 80),
        )

        draw_text(
            frame,
            f"Target: {current_expression}",
            (20, 120),
            size=1.0,
            color=(255, 255, 0),
        )

        target_image = resize_image(
            expression_images[current_expression],
            260,
            210,
        )

        display_png(
            frame,
            target_image,
            frame_width // 2,
            230,
        )

        draw_text(
            frame,
            f"Detected: {detected_expression}",
            (20, 170),
            color=(0, 255, 255),
        )

        draw_text(
            frame,
            f"Confidence: {detection_confidence:.2f}",
            (20, 205),
        )

        if detected_expression == "NO FACE":
            feedback = "Show your face"
            feedback_color = (0, 0, 255)
        elif stable_expression == "HOLD":
            feedback = "Hold the expression..."
            feedback_color = (0, 255, 255)
        elif is_correct_now:
            feedback = "CORRECT!"
            feedback_color = (0, 255, 0)
        else:
            feedback = "Try to imitate the target"
            feedback_color = (0, 0, 255)

        draw_text(
            frame,
            feedback,
            (frame_width // 2 - 170, frame_height - 90),
            color=feedback_color,
        )

        draw_text(
            frame,
            f"Score: {score}",
            (20, frame_height - 45),
            color=(0, 255, 255),
        )

        draw_text(
            frame,
            "Press Q to quit",
            (frame_width - 190, frame_height - 45),
            size=0.6,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if time.time() - round_start_time >= ROUND_DURATION:
            if round_correct:
                score += 1
                print(
                    f"Round {round_number}: CORRECT "
                    f"Target={current_expression}"
                )
            else:
                print(
                    f"Round {round_number}: INCORRECT "
                    f"Target={current_expression}, "
                    f"Detected={detected_expression}"
                )

            possible_expressions = [
                expression
                for expression in EXPRESSIONS
                if expression != current_expression
            ]

            current_expression = random.choice(
                possible_expressions
            )

            round_number += 1
            round_start_time = time.time()
            round_correct = False
            last_expression = "NO FACE"
            expression_streak = 0
            feature_history.clear()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()

print("Game ended.")
print("Final score:", score)