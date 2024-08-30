# Hand Sign Language to Text

This project aims to convert hand sign language gestures into text using computer vision techniques. The project utilizes MediaPipe for hand tracking, OpenCV for image processing, and machine learning algorithms to classify hand gestures. The best-performing model is then used to recognize gestures in real-time from a live camera feed.

## Project Overview

### 1. Data Collection (`imgs_gen.ipynb`)

- **Objective**: Collect a dataset of hand sign language gestures.
- **Method**: We captured images of different hand signs using a webcam. Each image corresponds to a specific hand gesture representing a letter or word.
- **Output**: A collection of images categorized by the corresponding hand sign.

### 2. Data Preprocessing (`data_preprocessing.ipynb`)

- **Objective**: Extract meaningful features from the hand sign images.
- **Method**: 
  - We used MediaPipe to detect and track hand landmarks.
  - The hand points (landmarks) extracted from each image were saved into a pickle file (`.p`), creating a structured dataset ready for model training.
- **Output**: A pickle file containing the preprocessed hand landmarks data.

### 3. Model Training (`training.ipynb`)

- **Objective**: Train machine learning models to classify hand gestures.
- **Method**: 
  - We trained three different models (Random Forest, XGBoost, and a TensorFlow neural network) using the preprocessed data.
  - The models were evaluated based on accuracy and other relevant metrics.
  - The best-performing model was selected and saved in a pickle file (`model.p`).
- **Output**: A trained model saved in `model.p`.

### 4. Real-Time Testing (`testing.ipynb`)

- **Objective**: Test the trained model on live camera feed.
- **Method**: 
  - We integrated the trained model with a real-time camera feed using OpenCV.
  - The live hand gestures captured by the camera were processed using MediaPipe, and the model predicted the corresponding text in real-time.
- **Output**: Real-time hand gesture recognition displayed as text on the screen.

## Getting Started

### Prerequisites

- Python 3.7+
- MediaPipe
- OpenCV
- Scikit-learn
- XGBoost
- TensorFlow

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hand-sign-language-to-text.git
