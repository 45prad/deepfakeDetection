

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load the LSTM model
model = load_model(r'9K_lstm_features.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to extract MFCC features
def extract_features(file_name, n_mfcc=13):
    audio, sample_rate = librosa.load(file_name)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Preprocess audio for model prediction
def preprocess_audio(file_path):
    mfccs_scaled = extract_features(file_path)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=0)  # Shape: (1, 13)
    mfccs_scaled = np.expand_dims(mfccs_scaled, axis=1)  # Shape: (1, 1, 13)
    return mfccs_scaled

@app.route('/')
def index():
    return render_template('audiopred.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        audio_features = preprocess_audio(filepath)
        prediction = model.predict(audio_features)
        result = "Bonafide" if prediction[0][0] > 0.5 else "Spoof"
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
