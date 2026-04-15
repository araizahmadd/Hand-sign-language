"""
config.py — Central configuration for the Hand Sign Language project.

All paths, constants, and label mappings live here. Import from this
module in notebooks and scripts to avoid hard-coding values.
"""

import os
from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"

# ── Artifact file paths ────────────────────────────────────────────────────────
PICKLE_PATH = MODELS_DIR / "data.pickle"
MODEL_PATH  = MODELS_DIR / "model.p"

# ── Data collection settings ───────────────────────────────────────────────────
NUM_CLASSES  = 3        # Number of gesture classes to collect
DATASET_SIZE = 150      # Images per class
CAMERA_INDEX = 0        # Webcam device index (change if using external cam)

# ── Label mapping ──────────────────────────────────────────────────────────────
LABELS_DICT: dict[int, str] = {
    0: "Hello",
    1: "Yes",
    2: "I Love You",
}

# ── MediaPipe settings ─────────────────────────────────────────────────────────
MP_DETECTION_CONFIDENCE = 0.3
MP_STATIC_IMAGE_MODE    = True     # True for preprocessing; False for real-time

# ── Training settings ──────────────────────────────────────────────────────────
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
NN_EPOCHS      = 50
NN_BATCH_SIZE  = 32
NN_VAL_SPLIT   = 0.2

# ── Ensure output directories exist ───────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
