from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained MobileNet model
model = MobileNetV2(weights='imagenet')

# Function to process the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image)          # Convert to numpy array
    image = preprocess_input(image)  # Preprocess for MobileNet
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Get the uploaded file
    file = request.files['file']
    try:
        # Open the image file
        image = Image.open(file)
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image)
        decoded = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions

        # Format the predictions
        results = [{"label": label, "probability": float(prob)} for (_, label, prob) in decoded]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
