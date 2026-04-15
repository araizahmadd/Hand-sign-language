"""
train.py — Train a Random Forest classifier on preprocessed hand landmarks.

Loads models/data.pickle, trains a RandomForestClassifier, evaluates it,
and saves the trained model to models/model.p.

Usage:
    python src/train.py
"""

import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    MODEL_PATH,
    PICKLE_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and return (data, labels) arrays from the pickle file."""
    if not os.path.exists(PICKLE_PATH):
        raise FileNotFoundError(
            f"Pickle file not found at {PICKLE_PATH}. "
            "Run `python src/preprocess.py` first."
        )
    with open(PICKLE_PATH, "rb") as f:
        data_dict = pickle.load(f)

    return np.asarray(data_dict["data"]), np.asarray(data_dict["labels"])


def train() -> None:
    print("Loading preprocessed data...")
    data, labels = load_data()
    print(f"  Samples: {len(data)}  |  Feature dim: {data.shape[1]}  |  Classes: {np.unique(labels)}\n")

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test,    y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test,        y_pred, average="weighted", zero_division=0)

    print("\n── Evaluation Results ───────────────────────────────────")
    print(f"  Accuracy  : {accuracy  * 100:.2f}%")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("\n── Classification Report ────────────────────────────────")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model}, f)

    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
