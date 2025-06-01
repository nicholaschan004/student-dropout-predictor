import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Roadmap:
# 1. Load and preprocess data
# 2. Feature selection and engineering
# 3. Train/test split
# 4. Model training and evaluation
# 5. Save model and scaler for deployment

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select important features
    features = [
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (enrolled)',
        'Tuition fees up to date',
        'Curricular units 2nd sem (evaluations)',
        'Age at enrollment',
        'Unemployment rate'
    ]
    
    # Select target variable
    target = 'Target'
    
    # Prepare X and y
    X = df[features].copy()
    y = df[target].copy()
    
    return X, y

def train_model():
    # Load and preprocess data
    X, y = load_and_preprocess_data('student_dropout.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('\nModel Performance:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('\nDetailed Classification Report:')
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print('\nModel and scaler saved successfully!')

if __name__ == '__main__':
    # Delete any existing testing models
    import os
    if os.path.exists('logistic_regression_model (1).pkl'):
        os.remove('logistic_regression_model (1).pkl')
    if os.path.exists('mock_model.py'):
        print('Removing mock model file')
    
    train_model()