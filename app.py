# app.py
# Description: This script creates a Flask web application to serve the
# trained diabetes prediction model. It provides an API endpoint to
# receive patient data and return a prediction with a confidence score.

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Scaler ---
# We use a global variable to load the model only once when the app starts.
try:
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model or scaler not found. Please run main.py to train and save them first.")
    model = None
    scaler = None

# --- Application Routes ---

@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    """
    Handle prediction requests from the web form.
    Validates input data, scales it, and returns the model's prediction.
    """
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        patient_data = request.get_json()

        # Validate that all required fields are present
        required_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(feature in patient_data for feature in required_features):
            return jsonify({'error': 'Missing one or more required fields.'}), 400

        # The order of features must match the training data
        feature_values = [
            float(patient_data['Pregnancies']),
            float(patient_data['Glucose']),
            float(patient_data['BloodPressure']),
            float(patient_data['SkinThickness']),
            float(patient_data['Insulin']),
            float(patient_data['BMI']),
            float(patient_data['DiabetesPedigreeFunction']),
            float(patient_data['Age'])
        ]

        # Convert to a numpy array for scaling
        final_features = np.array(feature_values).reshape(1, -1)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_probability = model.predict_proba(scaled_features)

        # Get the confidence score for the predicted class
        confidence_score = prediction_probability[0][int(prediction[0])]

        # Return the result as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'confidence': float(confidence_score)
        })

    except (ValueError, TypeError) as e:
        print(f"Error during data conversion: {e}")
        return jsonify({'error': 'Invalid input data. Please ensure all values are numbers.'}), 400
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(debug=True)
