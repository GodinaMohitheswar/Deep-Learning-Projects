# Image-Classification

# Overview
This project implements an image classification model using deep learning with TensorFlow and Keras. The model distinguishes between images of cats and dogs using the VGG16 architecture with transfer learning.

# Dataset
The dataset used is the "Dogs vs Cats" dataset from Kaggle, which contains images of cats and dogs for training and testing.

# Dataset Details
Training Data: 20,000 images
Validation Data: 5,000 images
Classes: 2 (Cat and Dog)
# Technologies Used
Python
TensorFlow
Keras
OpenCV
Matplotlib
# Project Structure
1. Image_Classification.ipynb: Python script implementing the image classification model.
- Data preprocessing: ImageDataGenerator for data augmentation.
- Model creation: VGG16 base model with custom top layers for classification.
- Training: Training the model on the dataset.
- Fine-tuning: Fine-tuning the model by unfreezing selected layers.
- Evaluation: Displaying training and validation accuracy and loss.
- Prediction: Function to predict the class of new images.
- cat.jpg and dog.jpg: Example images for testing the model.
2. README.md: This file, providing an overview of the project.
# Install Required Libraries:
pip install tensorflow keras opencv-python matplotlib
# Model Performance
After training and fine-tuning, the model achieved the following accuracy:
- Training Accuracy: 97.94%
- Validation Accuracy: 97.84%
