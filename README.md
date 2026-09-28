# Apple & Banana Detector

A Flask web app that detects apples and bananas in uploaded images using a custom-trained YOLOv8s model. Upload a photo, and the app draws labelled bounding boxes around each detected fruit.

## Features

- Upload `.png`, `.jpg` or `.jpeg` images through a simple web form
- Object detection with a YOLOv8s model trained on two classes: `apple` (0) and `banana` (1)
- Bounding boxes and confidence scores drawn on the image, plus a list of what was found (detections below 0.5 confidence are hidden)
- Runs on GPU when CUDA is available, otherwise on CPU
- Uploads get a random prefix so files with the same name don't overwrite each other
- Docker and Render deployment configs included

## Project Structure

```
.
├── app.py                       # Flask app: routes and upload handling
├── model_utils.py               # Loads the model and runs detection
├── apple_banana_detector.pth    # Trained YOLOv8s weights (state_dict)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image definition
├── render.yaml                  # Render.com deployment config
├── templates/
│   └── index.html               # Upload page and results view
└── static/
    ├── styles.css               # Page styling
    └── uploads/                 # Uploaded/processed images and the dataset
        ├── train/  (168 images + YOLO labels)
        ├── valid/  (48 images + YOLO labels)
        └── test/   (24 images + YOLO labels)
```

## Dataset

The dataset lives in `static/uploads/{train,valid,test}`, each with `images/` and `labels/` folders. Labels use the YOLO format, one object per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised to 0–1. Class `0` is apple and class `1` is banana.

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone <your-repo-url>
cd "module 7 part 2"

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Running Locally

```bash
python app.py
```

Then open http://localhost:5000 in your browser, choose an image and click **Detect Fruit**.

Set `FLASK_DEBUG=1` to enable debug mode, or `PORT` to use a different port.

## Running with Docker

```bash
docker build -t apple-banana-detector .
docker run -p 5000:5000 apple-banana-detector
```

## Deploying to Render

`render.yaml` defines a Docker-based web service called `apple-banana-detector`. Connect the repository to [Render](https://render.com) and it will build from the `Dockerfile` automatically. The app listens on the `PORT` that Render provides.

## How It Works

1. The user uploads an image, which is saved to `static/uploads/`.
2. `model_utils.load_model()` rebuilds the YOLOv8s architecture with 2 classes and loads the weights from `apple_banana_detector.pth` (a plain PyTorch `state_dict`).
3. `model_utils.detect_objects()` runs the model. Ultralytics handles resizing and scales the boxes back to the original image size.
4. Boxes scoring 0.5 or higher are drawn on the image (red for apples, yellow for bananas).
5. The annotated image is saved as `processed_<filename>` and shown next to the original, with a list of detections.

## Tech Stack

- [Flask](https://flask.palletsprojects.com/) for the web server
- [PyTorch](https://pytorch.org/) and [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for detection
- [OpenCV](https://opencv.org/) for image processing
