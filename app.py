from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import base64
import numpy as np

# Load the trained model
model = YOLO('runs/detect/guardrail_detector/weights/best.pt')
print("Model loaded successfully! ✅")

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

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