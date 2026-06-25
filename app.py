import os
import torch
from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np

# Load model directly with torch
model_path = 'runs/detect/guardrail_detector/weights/best.pt'
model = torch.hub.load('ultralytics/yolov8', 'custom', path=model_path, force_reload=False)
print("Model loaded successfully! ✅")

# Create Flask app
app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# Detection endpoint
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Run detection
    results = model(image)

    # Process results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })

    # Draw boxes
    annotated = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "detections": detections,
        "annotated_image": image_base64
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)