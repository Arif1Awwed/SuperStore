import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
# This object will handle routing and serving static files / templates
app = Flask(__name__)

# Define the path to the pre-trained scikit-learn/XGBoost pipeline model.
# Using os.path.dirname(__file__) ensures relative paths work correctly on any machine.
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    # Open and load the pickle file containing the trained Pipeline object.
    # We do this globally at startup so that we only read from disk once.
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    # If the model fails to load, set it to None and log the error.
    # The application will still start, but prediction requests will return a 500 error.
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    """
    Route handler for the home page.
    Renders and serves the templates/index.html file.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint that accepts input features from the frontend form,
    formats them into a Pandas DataFrame, passes them to the machine learning pipeline,
    and returns the predicted discount percentage.
    """
    # Safety check: Verify the model loaded successfully during application startup
    if model is None:
        return jsonify({"error": "Model is not loaded on the server."}), 500

    try:
        # Parse the JSON payload sent by the frontend AJAX request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract features from the payload and perform standard type casting.
        # This ensures strings are sanitised and numerical figures are mathematical types.
        input_data = {
            'Ship Mode': str(data.get('Ship Mode', 'Standard Class')),
            'Segment': str(data.get('Segment', 'Consumer')),
            'City': str(data.get('City', '')),
            'State': str(data.get('State', '')),
            'Region': str(data.get('Region', 'Central')),
            'Category': str(data.get('Category', '')),
            'Sub-Category': str(data.get('Sub-Category', '')),
            'Sales': float(data.get('Sales', 0.0)),
            'Quantity': int(data.get('Quantity', 1)),
            'Profit': float(data.get('Profit', 0.0))
        }

        # Create a single-row Pandas DataFrame using the keys from input_data.
        # The model's pipeline was trained using DataFrame inputs, not raw arrays.
        df = pd.DataFrame([input_data])
        
        # IMPORTANT: Scikit-learn pipelines require columns to be in the exact order
        # they were presented during training. model.feature_names_in_ contains this order.
        # This step reorders our DataFrame columns accordingly.
        df = df[model.feature_names_in_]

        # Execute the model's pipeline to predict the target (Discount)
        # model.predict returns a numpy array of predictions; we grab the first element.
        prediction = model.predict(df)
        predicted_discount = float(prediction[0])

        # Post-Processing: Clip the predicted value to the valid mathematical range [0.0, 1.0].
        # Since regressor outputs are continuous, extreme input metrics might lead to slightly
        # negative predictions (e.g. -0.01) which we clip back to 0.0 (0%).
        predicted_discount = max(0.0, min(1.0, predicted_discount))

        # Return the final calculated discount to the frontend client
        return jsonify({
            "predicted_discount": predicted_discount,
            "success": True
        })

    except ValueError as ve:
        # Handled if float() or int() casting fails due to malformed values
        return jsonify({"error": f"Invalid numerical inputs: {ve}"}), 400
    except Exception as e:
        # Standard safety catch for other runtime exceptions
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask development server on the local loopback interface (port 5000).
    # debug=True allows automatic reloading when code files are modified.
    app.run(debug=True, host='127.0.0.1', port=5000)
