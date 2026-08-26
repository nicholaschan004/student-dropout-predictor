"""Flask front end for the dropout classifier.

The model and scaler are loaded at import time rather than inside a
`__main__` block, so this module works unchanged under a WSGI server
(gunicorn, Vercel) and not only under `python app.py`.
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

from config import CLASS_LABELS, FEATURES, MODEL_PATH, SCALER_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

model = joblib.load(os.path.join(BASE_DIR, MODEL_PATH))
scaler = joblib.load(os.path.join(BASE_DIR, SCALER_PATH))

# Maps the JSON keys the form posts onto the column names the scaler was fitted
# with, and states the range each one is allowed to take. Anything outside these
# is a typo rather than a student, and extrapolating on it would be dishonest.
FIELDS = {
    "units_approved": ("Curricular units 2nd sem (approved)", 0, 30),
    "units_grade": ("Curricular units 2nd sem (grade)", 0, 20),
    "units_enrolled": ("Curricular units 2nd sem (enrolled)", 0, 30),
    "tuition_up_to_date": ("Tuition fees up to date", 0, 1),
    "units_evaluations": ("Curricular units 2nd sem (evaluations)", 0, 40),
    "age": ("Age at enrollment", 16, 80),
    "unemployment_rate": ("Unemployment rate", 0, 30),
}


class InvalidInput(Exception):
    pass


def parse_features(payload):
    """Turn the posted JSON into a one row DataFrame, or raise InvalidInput."""
    if not isinstance(payload, dict):
        raise InvalidInput("expected a JSON object")

    row = {}
    for key, (column, low, high) in FIELDS.items():
        if key not in payload:
            raise InvalidInput(f"missing field: {key}")
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            raise InvalidInput(f"{key} must be a number")
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidInput(f"{key} must be a finite number")
        if not low <= value <= high:
            raise InvalidInput(f"{key} must be between {low} and {high}")
        row[column] = value

    # Column order matters: the scaler and model were fitted on FEATURES.
    return pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/results")
def results():
    return render_template("results.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_LABELS})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = parse_features(request.get_json(silent=True))
    except InvalidInput as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    scaled = pd.DataFrame(scaler.transform(features), columns=FEATURES)
    probabilities = model.predict_proba(scaled)[0]

    # argmax of predict_proba rather than a second predict() call, so the label
    # shown can never disagree with the bar that is tallest in the chart.
    index = int(probabilities.argmax())

    return redirect(
        url_for(
            "results",
            prediction=CLASS_LABELS[index],
            confidence=round(float(probabilities[index]) * 100, 2),
            probabilities=",".join(f"{p:.6f}" for p in probabilities),
        )
    )


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"},
        port=int(os.environ.get("PORT", 8080)),
    )
