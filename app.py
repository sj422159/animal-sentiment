from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import cv2
import numpy as np
from collections import Counter
import os
import tempfile
import base64
from io import BytesIO



app = Flask(__name__)

# === Class Labels ===
classes = ['angry', 'happy', 'relaxed', 'sad']

# === Device Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Model Setup ===
model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load('efficientnet_b0_dog_sentiment.pth', map_location=device))
model = model.to(device)
model.eval()

# === Transformation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Routes ===
def pil_to_b64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    image_pred = None
    video_pred = None
    sampled_frames = []  # List of base64 strings
    sampled_labels = []  # List of predicted labels

    if request.method == 'POST':
        # === Image Upload Handling ===
        image_file = request.files.get('image_file')
        if image_file and image_file.filename != '':
            try:
                img = Image.open(image_file).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    pred_idx = torch.argmax(output, 1).item()
                    image_pred = classes[pred_idx]
            except Exception as e:
                image_pred = f"Error: {e}"

        # === Video Upload Handling ===
        video_file = request.files.get('video_file')
        if video_file and video_file.filename != '':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                video_file.save(tmpfile.name)
                temp_video_path = tmpfile.name

            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                video_pred = "Error opening video"
            else:
                predicted_labels = []
                frame_id = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_indices = np.linspace(0, total_frames - 1, 10, dtype=int)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_id in sample_indices:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        b64_img = pil_to_b64(pil_img)
                        sampled_frames.append(b64_img)

                        input_tensor = transform(pil_img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(input_tensor)
                            pred_idx = torch.argmax(output, 1).item()
                            label = classes[pred_idx]
                            sampled_labels.append(label)

                    frame_id += 1

                cap.release()
                os.unlink(temp_video_path)

                if sampled_labels:
                    counter = Counter(sampled_labels)
                    most_common_label = counter.most_common(1)[0][0]
                    video_pred = most_common_label
                else:
                    video_pred = "No valid frames"

    return render_template(
        'upload.html',
        image_pred=image_pred,
        video_pred=video_pred,
        sampled_frames=sampled_frames,
        sampled_labels=sampled_labels
    )
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame received'}), 400

    file = request.files['frame']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f"Invalid image: {str(e)}"}), 400

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, 1).item()
        label = classes[pred_idx]

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)