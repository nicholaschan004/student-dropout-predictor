# Student Dropout Prediction

This project uses machine learning to predict whether a student will drop out, remain enrolled, or graduate based on various academic and demographic factors.

## Features

- Web-based interface for inputting student data
- Machine learning model trained on student dropout dataset
- Prediction results with confidence scores
- Visualization of prediction probabilities

## Dataset

The model is trained on the `student_dropout.csv` dataset, which includes the following features:

- Curricular units approved in 2nd semester
- Curricular units grade in 2nd semester
- Curricular units enrolled in 2nd semester
- Tuition fees status
- Curricular units evaluations in 2nd semester
- Age at enrollment
- Unemployment rate

## Model

The prediction model is a logistic regression classifier that achieves approximately 77% accuracy and an F1 score of 0.75 on the student dropout dataset.

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:8080/`

## Files

- `app.py`: Flask web application
- `train_model.py`: Script to train the logistic regression model
- `model.joblib`: Trained model file
- `scaler.joblib`: Feature scaler
- `templates/`: HTML templates for the web interface
- `requirements.txt`: Required Python packages

## Usage

1. Enter the student's academic and demographic information in the web form
2. Click "Predict" to get the prediction result
3. View the prediction result and confidence score on the results page