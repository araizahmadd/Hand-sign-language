"""
collect_data.py — Webcam-based image collection for hand sign gestures.

Usage:
    python src/collect_data.py
    python src/collect_data.py --classes 5 --size 200

Each class will be saved under data/<class_index>/.
Press 'Q' when the prompt appears to start capturing a class.
"""

import argparse
import os
import sys

import cv2

# ── Allow running as a script from root ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    CAMERA_INDEX,
    DATA_DIR,
    DATASET_SIZE,
    LABELS_DICT,
    NUM_CLASSES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture hand gesture images from webcam."
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=NUM_CLASSES,
        help=f"Number of gesture classes to collect (default: {NUM_CLASSES})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DATASET_SIZE,
        help=f"Number of images per class (default: {DATASET_SIZE})",
    )
    return parser.parse_args()


def collect(num_classes: int, dataset_size: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam at index {CAMERA_INDEX}. "
            "Try changing CAMERA_INDEX in src/config.py."
        )

    for class_idx in range(num_classes):
        class_dir = os.path.join(DATA_DIR, str(class_idx))
        os.makedirs(class_dir, exist_ok=True)

        label_name = LABELS_DICT.get(class_idx, str(class_idx))
        print(f"\n[{class_idx + 1}/{num_classes}] Collecting data for: '{label_name}'")
        print("  → Get into position, then press 'Q' to start capturing.")

        # ── Wait for user to be ready ──────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(
                frame,
                f"Class: {label_name}  |  Press Q to start",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Hand Sign — Data Collection", frame)
            if cv2.waitKey(20) == ord("q"):
                break

        # ── Capture frames ─────────────────────────────────────────────────────
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(
                frame,
                f"Capturing {label_name}: {counter + 1}/{dataset_size}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Hand Sign — Data Collection", frame)
            cv2.waitKey(20)
            cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
            counter += 1

        print(f"  ✓ Saved {counter} images to {class_dir}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")


if __name__ == "__main__":
    args = parse_args()
    collect(args.classes, args.size)
