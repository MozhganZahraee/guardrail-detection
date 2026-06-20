import os
from flask import jsonify, Flask,request
from ultralytics import YOLO
import cv2
import base64
import numpy as np

# ***  1. Load the trained model ***
model = YOLO('runs/detect/guardrail_detector/weights/best.pt')
print("Model loaded successfully! ✅")

# ── 2. Create Flask app ──
app = Flask(__name__)

# ── 3. Health check endpoint ──
@app.route('/health',methods = ['GET']) 

def health():
    return jsonify({'status' : 'ok'})

# ── 4. Detection endpoint ──
@app.route('/detect', methods = ['POST'])

def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    image = np.frombuffer(file.read(), np.int8)
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    
    # Run detection
    results = model(image)
    
    # Process results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })
    
    # Draw boxes on image
    annotated = results[0].plot()
    
    # Convert to base64 to send back
    _, buffer   = cv2.imencode('.jpg', annotated)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "detections"     : detections,
        "annotated_image": image_base64
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

    
    