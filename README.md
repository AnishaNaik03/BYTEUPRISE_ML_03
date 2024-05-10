# BYTEUPRISE_ML_03
# Cat and Dog Image Classification with Support Vector Machine (SVM)
This project implements an image classification model using Support Vector Machine (SVM) to categorize images of cats and dogs. The SVM model is trained on the Kaggle dataset containing labeled images of cats and dogs, enabling accurate classification of new images.

# Dataset
Dataset URL: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

# Project Overview
The project includes the following steps:
* Data Preparation:
Images are loaded using OpenCV (cv2) and resized to 50x50 pixels.
Images are flattened to create feature vectors for SVM training.
* Model Training:
The SVM model is trained using the sklearn.svm.SVC class with a polynomial kernel (kernel='poly') and default parameters.
Training and testing data are split using train_test_split from sklearn.model_selection.
* Model Evaluation:
The trained model's accuracy is evaluated using a test set, and predictions are made on new images.

# Requirements
Ensure you have the following Python packages installed (see requirements.txt):
* numpy
* pandas
* matplotlib
* opencv-python
* scikit-learn
