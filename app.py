from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# Global variables
dataset = None
model = None

# Path to save the trained model
MODEL_PATH = 'model.pkl'

# Load pre-trained model if it exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    """Root endpoint to display a welcome message."""
    return "Welcome to the Manufacturing API! Use /upload, /train, or /predict endpoints."

@app.route('/favicon.ico')
def favicon():
    """Handle /favicon.ico requests to suppress errors."""
    return '', 204

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload a CSV dataset."""
    global dataset
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded!"}), 400

    try:
        # Load the dataset into a DataFrame
        dataset = pd.read_csv(file)
        return jsonify({"message": "Dataset uploaded successfully!", "columns": list(dataset.columns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train the Random Forest model."""
    global dataset, model
    if dataset is None:
        return jsonify({"error": "No dataset uploaded yet!"}), 400

    try:
        # Define features and target variable
        X = dataset[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
        y = dataset['Machine failure']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest Classifier
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)

        # Save the trained model to disk
        joblib.dump(model, MODEL_PATH)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        return jsonify({"message": "Model trained successfully!", "accuracy": round(accuracy, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction using the trained model."""
    global model
    if model is None:
        return jsonify({"error": "Model is not trained yet!"}), 400

    try:
        # Parse JSON input
        data = request.json
        features = [[
            data['Air temperature [K]'],
            data['Process temperature [K]'],
            data['Rotational speed [rpm]'],
            data['Torque [Nm]'],
            data['Tool wear [min]']
        ]]

        # Make prediction and calculate confidence
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])

        return jsonify({
            "Machine Failure": "Yes" if prediction == 1 else "No",
            "Confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
