from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.losses import Loss

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model from the same directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnnImagedeepfakemodel.h5")

try:
    # Attempt to load the model
    MODEL = tf.keras.models.load_model(MODEL_PATH)

    # Patch the loss function if necessary
    if hasattr(MODEL, 'loss') and isinstance(MODEL.loss, Loss):
        MODEL.loss.reduction = 'sum_over_batch_size'  # Set a valid reduction value

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None

CLASS_NAMES = ["fake", "real"]

def preprocess_image(image):
    # Resize the image to the target size expected by the model
    image = image.resize((256, 256))
    # Convert the image to a numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)
    # Expand dimensions to match the model's input shape
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model failed to load. Please check the logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Make a prediction
        prediction = MODEL.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = round(100 * (np.max(prediction[0])), 2)

        # Return the prediction and confidence
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)