from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.resnext = models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V1')
        self.resnext = nn.Sequential(*list(self.resnext.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(2048, 2048, batch_first=True)
        self.linear = nn.Linear(2048, 1)

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.resnext(x)
        features = self.adaptive_pool(features)
        features = torch.flatten(features, start_dim=1)
        features = features.view(batch_size, num_frames, -1)
        lstm_output, _ = self.lstm(features)
        lstm_output = lstm_output[:, -1, :]
        output = self.linear(lstm_output)
        return output

model = DeepFakeDetector()
model.load_state_dict(torch.load(r'C:\deepfake\models\dfd_model_weights.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_video(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None
    
    if len(frames) >= num_frames:
        frame_indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
        selected_frames = [frames[i] for i in frame_indices]
    else:
        selected_frames = frames + [frames[-1]] * (num_frames - len(frames))

    transformed_frames = [transform(frame) for frame in selected_frames]
    return torch.stack(transformed_frames)

def predict_video(video_path, threshold=0.35):
    video_tensor = preprocess_video(video_path)
    if video_tensor is None:
        return None, None
    video_tensor = video_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(video_tensor)
        outputs = torch.sigmoid(outputs).item()
        prediction = "Fake" if outputs >= threshold else "Real"
    return prediction, outputs * 100

@app.route('/')
def index():
    return render_template('videopred.html')

@app.route('/predict_video', methods=['POST'])
def predict_video_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    prediction, confidence = predict_video(filepath)
    if prediction is None:
        return jsonify({'error': 'Invalid video file'}), 400
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
