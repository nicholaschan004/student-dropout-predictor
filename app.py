from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

app = Flask(__name__)

# Initialize these as None, they will be loaded when the model is trained
model = None
scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features from the form data
        features = [
            float(data['units_approved']),  # Curricular units 2nd sem (approved)
            float(data['units_grade']),     # Curricular units 2nd sem (grade)
            float(data['units_enrolled']),  # Curricular units 2nd sem (enrolled)
            float(data['tuition_up_to_date']),  # Tuition fees up to date
            float(data['units_evaluations']),  # Curricular units 2nd sem (evaluations)
            float(data['age']),            # Age at enrollment
            float(data['unemployment_rate'])  # Unemployment rate
        ]
        
        # Reshape and scale the features
        features = np.array(features).reshape(1, -1)
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        if model is not None:
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)[0]
            
            # Get the highest probability
            max_prob = max(probabilities)
            confidence = round(max_prob * 100, 2)
            
            # Map prediction to class label
            class_labels = ['Dropout', 'Enrolled', 'Graduate']
            predicted_class = class_labels[prediction[0]]
            
            # Redirect to results page with parameters
            from flask import redirect, url_for
            import json
            return redirect(url_for('results', 
                prediction=predicted_class,
                confidence=confidence,
                probabilities=json.dumps(probabilities.tolist())
            ))
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    # Load the logistic regression model and scaler
    if os.path.exists('model.joblib'):
        model = joblib.load('model.joblib')
    if os.path.exists('scaler.joblib'):
        scaler = joblib.load('scaler.joblib')
    
    app.run(debug=True, port=8080)