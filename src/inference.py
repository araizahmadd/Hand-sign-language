"""
inference.py — Real-time hand gesture recognition from webcam.

Loads the trained model from models/model.p, captures frames from the
webcam, detects hand landmarks with MediaPipe, and overlays the predicted
gesture label on the live video feed.

Usage:
    python src/inference.py
    python src/inference.py --camera 1   # if using an external webcam

Press 'Q' to quit.
"""

import argparse
import os
import pickle
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    CAMERA_INDEX,
    LABELS_DICT,
    MODEL_PATH,
    MP_DETECTION_CONFIDENCE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time hand sign recognition.")
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Webcam device index (default: {CAMERA_INDEX})",
    )
    return parser.parse_args()


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python src/train.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)["model"]


def run(camera_idx: int) -> None:
    model = load_model()
    print(f"Model loaded from {MODEL_PATH}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,   # streaming mode for lower latency
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
    )

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam at index {camera_idx}. "
            "Try --camera <index> with a different device number."
        )

    print("Running inference — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Exiting.")
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux: list[float] = []
            x_coords: list[float] = []
            y_coords: list[float] = []

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)

                min_x, min_y = min(x_coords), min(y_coords)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

            # Bounding box
            x1 = max(int(min(x_coords) * w) - 20, 0)
            y1 = max(int(min(y_coords) * h) - 20, 0)
            x2 = min(int(max(x_coords) * w) + 20, w)
            y2 = min(int(max(y_coords) * h) + 20, h)

            try:
                prediction = model.predict([np.asarray(data_aux)])
                label = LABELS_DICT.get(int(prediction[0]), str(prediction[0]))
            except Exception:
                label = "?"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 200, 100),
                3,
                cv2.LINE_AA,
            )

        # HUD overlay
        cv2.putText(
            frame,
            "Hand Sign Recognition | Q to quit",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    args = parse_args()
    run(args.camera)
