# Face Detection using Deep Learning

This repository contains a deep learning-based face detection model. The dataset was created manually, labeled using LabelMe, and augmented before training. The project includes model training, evaluation, and inference.

## Project Overview

- **Dataset**: Manually created, labeled with LabelMe, and augmented.
- **Preprocessing**: Data split into training, validation, and test sets.
- **Model**: Trained using TensorFlow/Keras.
- **Inference**: Uses the trained model to detect faces in images or video streams.

## Folder Structure

```
Face Detection/
│── logs/                     # Training logs
│── my_saved_model/           # Saved model directory
│── class_indices.npy         # Mapping of class labels
│── facedetection.ipynb       # Jupyter Notebook with training & inference code
│── my_model.keras            # Trained Keras model file
```

## Setup & Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/face-detection.git
   cd face-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook facedetection.ipynb
   ```

## Model Training

- The dataset was preprocessed and split into training, validation, and test sets.
- Data augmentation techniques were applied to improve model generalization.
- The model was trained using TensorFlow/Keras and saved in `.keras` format.
- Training logs were saved in the `logs/` directory.

## Inference

- Load the trained model from `my_model.keras`.
- Use the Jupyter Notebook to perform face detection on new images.
- The `class_indices.npy` file maps labels to their respective classes.

## Future Improvements

- Implement real-time face detection.
- Optimize the model for deployment on edge devices.


