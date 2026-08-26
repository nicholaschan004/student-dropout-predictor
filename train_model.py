"""Train the dropout classifier and write the artifacts the web app serves.

This is the only place a model is produced. `app.py` loads exactly what this
script writes, so what you evaluate here is what gets served.

    python train_model.py
"""

import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    CLASS_LABELS,
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df[FEATURES], df["Target"]


def train():
    X, y = load_data()

    # Stratified: "Enrolled" is only 18% of the data, and an unstratified split
    # leaves the test set with an unstable number of them.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit on the DataFrame, not a bare array, so the scaler records the feature
    # names and will complain at serve time if the columns ever drift.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (weighted): {f1:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))
    print("Confusion matrix (rows = true, cols = predicted)")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred), index=CLASS_LABELS, columns=CLASS_LABELS
    ))

    print("\nCoefficients per class (standardised units)")
    print(pd.DataFrame(model.coef_, index=CLASS_LABELS, columns=FEATURES).T.round(3))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    report = classification_report(
        y_test, y_pred, target_names=CLASS_LABELS, output_dict=True
    )
    with open(METRICS_PATH, "w") as fh:
        json.dump(
            {
                "accuracy": round(accuracy, 4),
                "f1_weighted": round(f1, 4),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "per_class": {
                    label: {k: round(v, 4) for k, v in report[label].items()}
                    for label in CLASS_LABELS
                },
            },
            fh,
            indent=2,
        )

    print(f"\nWrote {MODEL_PATH}, {SCALER_PATH}, {METRICS_PATH}")


if __name__ == "__main__":
    train()
