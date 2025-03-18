from flask import Flask, request, jsonify , render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained LSTM model
model = tf.keras.models.load_model('lstm_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to extract MFCC features from audio files
def extract_features(file_path, n_mfcc=13):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)  # Average over time steps
    return mfccs_scaled

# Function to preprocess audio for prediction
def preprocess_audio(file_path):
    # Extract the MFCC features (shape: (13,))
    mfccs_scaled = extract_features(file_path)

    # Reshape to match the model's expected input shape (1, 1, 13)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=0)  # Shape becomes (1, 13)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=1)  # Shape becomes (1, 1, 13)

    return mfccs_scaled
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
            file_path = os.path.join('uploads', file.filename)
            print(f"Saving file to: {file_path}")  # Log the file path
            file.save(file_path)

            # Verify the file exists after saving
            if not os.path.exists(file_path):
                print(f"File not saved: {file_path}")
                return jsonify({'error': 'File could not be saved'}), 500

            new_audio_features = preprocess_audio(file_path)
            prediction = model.predict(new_audio_features)
            result = "Bonafide" if prediction[0][0] > 0.5 else "Spoof"

            os.remove(file_path)  # Clean up after processing

            app.logger.info(f"Prediction: {result}")  # Log the prediction result
            return jsonify({'prediction': result})
        except Exception as e:
            app.logger.error(f"Error processing file: {e}")
            return jsonify({'error': 'Failed to process file'}), 500

    return render_template('index.html')  # Serve HTML file for upload form

if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
