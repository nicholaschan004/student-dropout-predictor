"""Contract shared by the trainer and the web app.

Both import from here so a change to the feature list or the class order can
never be applied to one side only. Keep this module free of heavy imports: the
serverless function loads it on every cold start.
"""

DATA_PATH = "student_dropout.csv"
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"
METRICS_PATH = "metrics.json"

# The seven columns the web form collects, in the order the model expects them.
FEATURES = [
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Tuition fees up to date",
    "Curricular units 2nd sem (evaluations)",
    "Age at enrollment",
    "Unemployment rate",
]

# The dataset ships the outcome pre-encoded, and NOT in alphabetical order, so
# this mapping cannot be inferred from the integers alone. Index i is the label
# for class i, matching model.classes_ == [0, 1, 2]. Verified against the data:
# class 1 is the largest group and has the highest mean units approved, which
# makes it Graduate rather than Enrolled.
CLASS_LABELS = ["Dropout", "Graduate", "Enrolled"]

RANDOM_STATE = 42
TEST_SIZE = 0.25
