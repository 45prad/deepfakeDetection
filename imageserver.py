from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import numpy as np
import os
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model (without compiling to avoid optimizer issues)
MODEL_PATH = 'cnnImagedeepfakemodel.h5'
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Recompile the model manually
MODEL.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Class labels
CLASS_NAMES = ["fake", "real"]

def predict(image):
    """Preprocess the image and make a prediction."""
    image = image.resize((256, 256))  # Resize to match model input size
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch input
    img_array /= 255.0  # Normalize to match model training
    
    predictions = MODEL.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('imagepred.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    """Handle file uploads and return predictions."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        try:
            image = Image.open(file).convert('RGB')  # Convert to RGB
            predicted_class, confidence = predict(image)
            return jsonify({'prediction': predicted_class, 'confidence': confidence})
        except Exception as e:
            app.logger.error(f"Error processing image: {e}")
            return jsonify({'error': 'Failed to process image'}), 500
    
    return render_template('index.html')  # Serve HTML file for upload form

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=8082, debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8020))
    app.run(host='0.0.0.0', port=port)