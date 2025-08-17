import cv2
import numpy as np

def detect_objects(model, image):
    # Preprocess image (resize, normalize)
    resized = cv2.resize(image, (224, 224))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)
    
    # Predict
    predictions = model.predict(input_tensor)
    
    # Post-process predictions (adjust for your model)
    boxes = predictions[0]  # Example: [x1, y1, x2, y2]
    scores = predictions[1]  # Confidence scores
    classes = predictions[2]  # 0=apple, 1=banana
    
    return boxes, scores, classes