import os
import uuid

import cv2
from flask import Flask, redirect, render_template, request
from werkzeug.utils import secure_filename

from model_utils import detect_objects, load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(os.path.join(BASE_DIR, 'apple_banana_detector.pth'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == '':
            return redirect(request.url)

        if not allowed_file(file.filename):
            return render_template('index.html', error='Please upload a PNG or JPG image.')

        # Prefix with a random id so uploads with the same name don't overwrite each other
        filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if cv2.imread(filepath) is None:
            os.remove(filepath)
            return render_template('index.html', error='That file could not be read as an image.')

        image, detections = detect_objects(model, filepath)

        processed = 'processed_' + filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], processed), image)

        return render_template('index.html',
                               original=filename,
                               processed=processed,
                               detections=detections)

    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
