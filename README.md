# GCP
Gender classification project

# Gender Classification using Deep Learning

This repository contains code for a gender classification model using deep learning. The model is built using TensorFlow and Keras libraries.

## Overview

The purpose of this project is to classify the gender of individuals based on their facial images. The model is trained on a dataset containing images of male and female faces.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Dataset

The dataset used for training, validation, and testing is organized in the following directory structure:

```
Dataset/
    ├── Train/
    │   ├── Male/
    │   └── Female/
    ├── Validation/
    │   ├── Male/
    │   └── Female/
    └── Test/
        ├── Male/
        └── Female/
```

## Model Architecture

The model architecture consists of:

- Transfer learning with VGG16 as the base model
- Fine-tuning the last few layers of VGG16
- Additional convolutional and dense layers
- Dropout regularization to prevent overfitting

## Training

The model is trained using an Adam optimizer with a binary cross-entropy loss function. Training includes data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping.

## Callbacks

Several callbacks are used during training:

- ReduceLROnPlateau: Reduces learning rate if validation loss plateaus
- ModelCheckpoint: Saves the best model during training
- EarlyStopping: Stops training early if validation loss does not improve

## Evaluation

The model's performance is evaluated using accuracy metrics on both training and validation sets. Additionally, a few sample images from the test set are classified to demonstrate model predictions.

## Results Visualization

The training and validation accuracies and losses are visualized using Matplotlib.

## Usage

1. Organize the dataset according to the specified directory structure.
2. Install the required dependencies.
3. Run the provided Python script.
4. Check the model's performance and predictions.

## Author

[Fahim]

## License

This project is licensed under the [MIT License](LICENSE).

---
