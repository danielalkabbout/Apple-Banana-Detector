from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from torchvision import transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('apple_banana_detector.pth', map_location=device)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image with PyTorch
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(tensor)
            
            # Process predictions here
            
            return render_template('index.html', result=filename)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0')