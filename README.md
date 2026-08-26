<div align="center">

# Student Dropout Predictor

*Predicting whether a university student drops out, graduates, or stays enrolled*

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Chart.js](https://img.shields.io/badge/Chart.js-4-ff6384?style=flat-square&logo=chartdotjs&logoColor=white)](https://www.chartjs.org)

[Overview](#overview) • [Results](#results) • [What the model learned](#what-the-model-learned) • [Getting started](#getting-started) • [Project layout](#project-layout)

</div>

A multinomial logistic regression trained on 3,682 university records, served behind a
Flask form that returns a class probability for each of the three outcomes.

The interesting part of this project is not the accuracy number. It is where the model
fails, and why the headline figure hides it.

## Overview

Universities want to know which students are at risk early enough to intervene. This model
uses seven signals available at the end of a student's second semester:

| Feature | Why it is in the model |
| --- | --- |
| Curricular units approved | How many courses were actually passed. Dominates every other signal. |
| Average grade | Performance in those courses, on the Portuguese 0 to 20 scale. |
| Curricular units enrolled | Course load taken on. |
| Curricular units evaluated | Assessments actually sat. |
| Tuition fees up to date | A financial distress signal rather than an academic one. |
| Age at enrollment | Older entrants follow different paths through a degree. |
| Unemployment rate | Macroeconomic conditions that year. |

## Results

Stratified 75/25 split, 2,761 training rows and 921 test rows, seed fixed at 42.
`python train_model.py` reproduces every number below and writes them to `metrics.json`.

**Accuracy 76.98%, weighted F1 0.751.**

That number on its own is misleading, so here is the per class breakdown:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Dropout | 0.83 | 0.76 | 0.79 | 294 |
| Graduate | 0.79 | 0.95 | 0.86 | 460 |
| **Enrolled** | **0.51** | **0.30** | **0.38** | 167 |

Confusion matrix, rows are the true class:

|  | Predicted Dropout | Predicted Graduate | Predicted Enrolled |
| --- | --- | --- | --- |
| **Dropout** | 223 | 43 | 28 |
| **Graduate** | 4 | 436 | 20 |
| **Enrolled** | 42 | 75 | 50 |

Dropout and Graduate separate cleanly, and the model almost never confuses one for the
other: only 4 graduates out of 460 were called dropouts. Enrolled is the problem. Seven in
ten students who were still enrolled got labelled as something else, and most of them were
called graduates.

That is a real limit of the feature set rather than a tuning issue. "Still enrolled after
two semesters" is not a distinct academic profile so much as an outcome that has not
resolved yet, and these seven features do not separate it from a slow graduate. Reporting
76.98% without this table would make the model look far more useful than it is.

## What the model learned

Coefficients are in standardised units, so they are directly comparable:

| Feature | Dropout | Graduate | Enrolled |
| --- | ---: | ---: | ---: |
| Curricular units approved | -1.580 | **2.178** | -0.598 |
| Average grade | -0.630 | 0.850 | -0.220 |
| Curricular units enrolled | 0.511 | -0.573 | 0.061 |
| Tuition fees up to date | -0.538 | 0.485 | 0.053 |
| Curricular units evaluated | -0.027 | -0.193 | 0.220 |
| Age at enrollment | 0.242 | -0.113 | -0.129 |
| Unemployment rate | 0.083 | -0.039 | -0.044 |

Units approved dominates everything else by a factor of three. Unemployment rate barely
registers, which is worth knowing before anyone spends effort sourcing macroeconomic data
for a future version.

One detail that is easy to get wrong: the dataset ships the outcome pre-encoded as 0/1/2
and **not** in alphabetical order, so the mapping cannot be guessed from the integers.
Class 1 is both the largest group and the one with the highest mean units approved, which
identifies it as Graduate. Getting that backwards silently swaps two classes in every
chart and coefficient table without raising a single error. The mapping lives in
`config.py` with the evidence for it written down next to it.

## Getting started

```bash
pip install -r requirements-dev.txt

python train_model.py     # trains, prints the report, writes model.joblib + metrics.json
python app.py             # serves on http://127.0.0.1:8080
```

`train_model.py` is the only thing that writes model artifacts, and `app.py` loads exactly
what it wrote, so what you see in the evaluation report is what gets served.

`requirements.txt` holds the runtime dependencies and `requirements-dev.txt` adds the
plotting libraries the notebook needs, so the deployed function does not ship matplotlib.

### API

```
GET  /health    -> {"status": "ok", "classes": [...]}
POST /predict   -> 302 to /results with the prediction and class probabilities
```

`POST /predict` validates every field against the range it is allowed to take and returns
a 400 with a specific message rather than extrapolating on a typo:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"units_approved":5,"units_grade":12.5,"units_enrolled":6,
       "tuition_up_to_date":1,"units_evaluations":8,"age":20,
       "unemployment_rate":11.1}'
```

## Project layout

```
config.py                    feature list and class order, shared by trainer and app
train_model.py               the only producer of model.joblib and scaler.joblib
app.py                       Flask app, loaded at import time so it runs under WSGI
api/index.py                 serverless entry point, re-exports the same app
logistic_regression.ipynb    the exploration that chose the feature set
templates/                   the form and the results page
student_dropout.csv          3,682 rows, 7 features, 1 pre-encoded target
```

`config.py` exists so the feature order and class labels cannot drift between the trainer
and the server. Both import from it; neither defines its own copy.

## Data

Derived from [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
in the UCI Machine Learning Repository, reduced to the seven features above. The dataset
covers Portuguese higher education students, which is why grades are on a 0 to 20 scale.

Predictions are statistical estimates over population averages. Nothing here should be
used to make a decision about an individual student.
