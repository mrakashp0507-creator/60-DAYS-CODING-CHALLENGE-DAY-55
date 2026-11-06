
import os
import csv
import io
import zipfile
import urllib.request

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -----------------------------------------
# 1. CONFIGURATION
# -----------------------------------------

PROCESSED_FILE = "processed_sms.csv"
MODEL_FILE = "spam_model.joblib"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/"
    "public/228/sms+spam+collection.zip"
)


# -----------------------------------------
# 2. LOAD DATASET
# -----------------------------------------

def load_dataset():

    if os.path.exists(PROCESSED_FILE):

        print("Loading Part 1 processed dataset...")

        data = pd.read_csv(PROCESSED_FILE)

        # Use original messages so useful signals such
        # as URLs, numbers and punctuation are retained.
        if "original_message" in data.columns:
            text_column = "original_message"
        elif "cleaned_message" in data.columns:
            text_column = "cleaned_message"
        else:
            raise ValueError(
                "No message column found in processed_sms.csv"
            )

        if "label" not in data.columns:
            raise ValueError("Dataset has no label column.")

        data = data[["label", text_column]].copy()
        data.columns = ["label", "message"]

    else:

        print("Processed file not found.")
        print("Downloading original UCI dataset...")

        request = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "Python Spam Detector"}
        )

        with urllib.request.urlopen(
            request, timeout=30
        ) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
            with archive.open("SMSSpamCollection") as file:

                content = file.read().decode(
                    "utf-8", errors="replace"
                )

        rows = list(csv.reader(
            io.StringIO(content),
            delimiter="\t"
        ))

        data = pd.DataFrame(
            rows,
            columns=["label", "message"]
        )

    data["label"] = data["label"].str.strip().str.lower()
    data["message"] = data["message"].fillna("").astype(str)

    data = data[
        data["label"].isin(["spam", "ham"])
        & (data["message"].str.strip() != "")
    ].copy()

    print("Dataset loaded successfully!")
    print("Total messages:", len(data))

    return data


# -----------------------------------------
# 3. PREPARE FEATURES AND LABELS
# -----------------------------------------

def prepare_data(data):

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining messages:", len(X_train))
    print("Testing messages:", len(X_test))

    return X_train, X_test, y_train, y_test


# -----------------------------------------
# 4. BUILD MACHINE LEARNING PIPELINE
# -----------------------------------------

def build_model():

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.5)
        )
    ])

    return model


# -----------------------------------------
# 5. TRAIN MODEL
# -----------------------------------------

def train_model(model, X_train, y_train):

    print("\nTraining machine learning model...")

    model.fit(X_train, y_train)

    print("Training completed!")

    return model


# -----------------------------------------
# 6. EVALUATE MODEL
# -----------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            target_names=["Not Spam", "Spam"],
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    )

    print("Confusion Matrix:")
    print(matrix)

    print("\nMatrix interpretation:")
    print("Rows = Actual labels")
    print("Columns = Predicted labels")
    print("Order = Not Spam, Spam")

    return accuracy


# -----------------------------------------
# 7. SAVE TRAINED MODEL
# -----------------------------------------

def save_model(model):

    joblib.dump(model, MODEL_FILE)

    print("\nTrained model saved as:", MODEL_FILE)


# -----------------------------------------
# 8. TEST CUSTOM MESSAGES
# -----------------------------------------

def test_custom_messages(model):

    messages = [
        "Congratulations! You have won a free prize. Claim now!",
        "Hey, are we meeting for lunch today?",
        "URGENT! You have won cash. Call now to claim your reward.",
        "Please call me when you reach home.",
        "Your account has been selected for a free gift. Click here!",
        "Can you send me the project notes?"
    ]

    print("\n" + "=" * 50)
    print("CUSTOM MESSAGE PREDICTIONS")
    print("=" * 50)

    for message in messages:

        prediction = model.predict([message])[0]

        if prediction == "spam":
            result = "SPAM"
        else:
            result = "NOT SPAM"

        probabilities = model.predict_proba([message])[0]

        spam_index = list(model.classes_).index("spam")
        spam_probability = probabilities[spam_index] * 100

        print("\nMessage:", message)
        print("Prediction:", result)
        print(f"Spam score: {spam_probability:.2f}%")


# -----------------------------------------
# 9. MAIN PROGRAM
# -----------------------------------------

def main():

    print("\nSMART SPAM DETECTOR - PART 2")
    print("-" * 50)

    try:

        data = load_dataset()

        X_train, X_test, y_train, y_test = (
            prepare_data(data)
        )

        model = build_model()

        model = train_model(
            model,
            X_train,
            y_train
        )

        evaluate_model(
            model,
            X_test,
            y_test
        )

        save_model(model)

        test_custom_messages(model)

        print("\nProject completed successfully!")

    except Exception as error:

        print("\nProject failed:", error)


if __name__ == "__main__":
    main()
