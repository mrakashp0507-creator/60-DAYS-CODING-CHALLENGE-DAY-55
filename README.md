
# 🛡️ Smart Spam Detector – Part 2

A machine-learning-based SMS spam classification system
using TF-IDF and Multinomial Naive Bayes.

## Objective

Train an ML model to automatically identify spam and
legitimate SMS messages.

## Features

- Load SMS dataset.
- Convert text into TF-IDF numerical vectors.
- Train a Multinomial Naive Bayes classifier.
- Evaluate model performance.
- Display accuracy, precision, recall and F1-score.
- Generate a confusion matrix.
- Predict custom SMS messages.
- Save the trained model using Joblib.

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Joblib

## Dataset

UCI SMS Spam Collection:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Installation

pip install -r requirements.txt

## Run

python train_model.py

## Machine Learning Workflow

1. Dataset loading
2. Train-test split
3. TF-IDF vectorization
4. Model training
5. Model evaluation
6. Custom message prediction
7. Model serialization

## Evaluation

The application reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Add your actual evaluation results after running
the program.

## Model

Algorithm: Multinomial Naive Bayes

Feature extraction: TF-IDF

Saved model: spam_model.joblib

## Future Improvements

- Try Logistic Regression and SVM.
- Add a graphical user interface.
- Build a live SMS prediction API.
- Improve detection of phishing links.
- Monitor false positives and false negatives.

## Project Status

Part 1: Data collection and preprocessing completed.

Part 2: Machine learning classifier and evaluation.
