import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

CLASS_NAMES = {0: 'apple', 1: 'banana'}


def load_model(weights_path):
    """Build a 2-class YOLOv8s and load the trained weights into it.

    The .pth file is a plain state_dict saved from an ultralytics YOLO object,
    so its keys look like 'model.model.0.conv.weight'. We rebuild the matching
    architecture and strip the leading 'model.' so the keys line up.
    """
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}

    net = DetectionModel('yolov8s.yaml', nc=len(CLASS_NAMES), verbose=False)
    net.load_state_dict(state_dict)
    net.names = CLASS_NAMES
    net.eval()

    model = YOLO('yolov8s.yaml', task='detect')
    model.model = net
    return model


def detect_objects(model, image_path, conf=0.5):
    """Run detection on an image file.

    Returns the image with boxes drawn on it (BGR numpy array) and a list of
    detections as dicts with 'label', 'confidence' and 'box' (x1, y1, x2, y2).
    """
    result = model.predict(image_path, conf=conf, verbose=False)[0]

    detections = []
    for box, score, class_id in zip(result.boxes.xyxy.tolist(),
                                    result.boxes.conf.tolist(),
                                    result.boxes.cls.tolist()):
        detections.append({
            'label': CLASS_NAMES[int(class_id)],
            'confidence': score,
            'box': [int(v) for v in box],
        })

    image = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = (78, 90, 255) if det['label'] == 'apple' else (60, 201, 255)
        text = f"{det['label']} {det['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, text, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image, detections
