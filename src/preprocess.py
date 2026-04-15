"""
preprocess.py — Extract MediaPipe hand landmarks from collected images.

Reads images from data/<class_index>/, detects hand landmarks using MediaPipe,
normalises them relative to the hand bounding box, and saves the result to
models/data.pickle.

Usage:
    python src/preprocess.py
"""

import os
import pickle
import sys

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    DATA_DIR,
    MP_DETECTION_CONFIDENCE,
    MP_STATIC_IMAGE_MODE,
    PICKLE_PATH,
)


def extract_landmarks(image_path: str, hands) -> list[float] | None:
    """
    Run MediaPipe on a single image and return normalised (x, y) landmark coords.

    Returns None if no hand is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Cannot read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    data_aux: list[float] = []

    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_coords), min(y_coords)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

    return data_aux


def preprocess() -> None:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=MP_STATIC_IMAGE_MODE,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
    )

    data: list[list[float]] = []
    labels: list[str] = []
    skipped = 0

    class_dirs = sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    )

    if not class_dirs:
        raise FileNotFoundError(
            f"No class directories found in {DATA_DIR}. "
            "Run `python src/collect_data.py` first."
        )

    print(f"Found {len(class_dirs)} class(es) in {DATA_DIR}\n")

    for class_dir in class_dirs:
        class_path = os.path.join(DATA_DIR, class_dir)
        images = [f for f in os.listdir(class_path) if f.endswith((".jpg", ".png"))]
        print(f"  Class '{class_dir}': {len(images)} images")

        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            landmarks = extract_landmarks(img_path, hands)

            if landmarks is None:
                skipped += 1
                continue

            data.append(landmarks)
            labels.append(class_dir)

    hands.close()

    print(f"\nExtracted landmarks from {len(data)} images ({skipped} skipped — no hand detected).")

    # Pad sequences so all samples have the same feature length (required for numpy arrays)
    max_len = max(len(d) for d in data)
    data_padded = [d + [0.0] * (max_len - len(d)) for d in data]

    payload = {"data": data_padded, "labels": labels}
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved preprocessed dataset → {PICKLE_PATH}")


if __name__ == "__main__":
    preprocess()
