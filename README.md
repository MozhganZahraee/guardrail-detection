# Guardrail Detection System

A machine learning project that detects highway guardrails in road images using YOLOv8, MLflow, and Flask.

## Project Overview

This project implements an object detection system specifically trained to identify highway guardrails and trees in road images. It uses YOLOv8 (a state-of-the-art real-time object detector) with transfer learning on a custom guardrail dataset.

**Use Case:** Can be deployed on autonomous vehicles to detect road guardrails in real-time.

## Features

- **YOLOv8 Detection**: Real-time object detection using transfer learning
- **MLflow Tracking**: Experiment tracking with automatic metric logging
- **Flask API**: REST API for image upload and detection
- **Docker Containerization**: Easy deployment across different environments
- **Model Performance**: Achieved mAP50 of 0.595 on validation set

## Project Structure
guardrail-detection/

├── app.py                 # Flask API application

├── train.py              # Model training script

├── Dockerfile            # Docker configuration

├── requirements.txt      # Python dependencies

├── README.md            # This file

├── dataset/

│   ├── data.yaml        # Dataset configuration

│   ├── train/           # Training images (231 images)

│   ├── valid/           # Validation images (4 images)

│   └── test/            # Test images (4 images)

├── runs/

│   └── detect/

│       └── guardrail_detector/

│           └── weights/

│               ├── best.pt    # Best model weights

│               └── last.pt    # Last epoch weights

└── venv/                # Python virtual environment

## Installation

### Prerequisites
- Python 3.13+
- Docker (optional, for containerization)
- Git

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/MozhganZahraee/guardrail-detection.git
cd guardrail-detection
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the YOLOv8 model on the guardrail dataset:

```bash
source venv/bin/activate
python3 train.py
```

This will:
- Load the pretrained YOLOv8s model
- Fine-tune on the guardrail dataset for 20 epochs
- Log all metrics to MLflow
- Save best and last model weights

**Training Results:**
- Best mAP50: 0.595 (59.5% accuracy)
- Precision: 0.984 (very high confidence)
- Recall: 0.4 (detects 40% of guardrails)

### Running the Flask API

```bash
source venv/bin/activate
python3 app.py
```

The API will start on `http://localhost:5001`

**Endpoints:**

1. **Health Check**
```bash
curl http://localhost:5001/health
```
Response:
```json
{"status": "ok"}
```

2. **Detect Guardrails**
```bash
curl -X POST http://localhost:5001/detect \
     -F "image=@path/to/image.jpg"
```
Response:
```json
{
  "detections": [
    {
      "class": "highway-guardrail",
      "confidence": 0.95,
      "bbox": [120, 340, 520, 390]
    }
  ],
  "annotated_image": "base64_encoded_string"
}
```

### Docker Deployment

Build the Docker image:
```bash
docker build -t guardrail-detection .
```

Run the container:
```bash
docker run -p 5001:5001 guardrail-detection
```

The API will be available at `http://localhost:5001`

## MLflow Experiment Tracking

View training experiments:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open browser at `http://127.0.0.1:5000` to see:
- Training metrics (mAP50, precision, recall)
- Training parameters (epochs, batch size, learning rate)
- Model artifacts

## Model Architecture

**YOLOv8s (Small)**
- 130 layers
- 11,136,374 parameters
- Real-time inference (~90ms per image)

**Architecture Components:**
1. **Backbone (CSPDarknet)**: Extracts features from images
2. **Neck (FPN)**: Combines multi-scale features
3. **Head (Detect)**: Makes predictions (bounding boxes + classes)

## Dataset

- **Source**: Roboflow Universe - Highway Guardrail Detection
- **Classes**: 2 (highway-guardrail, tree)
- **Total Images**: 239
  - Train: 231 images
  - Validation: 4 images
  - Test: 4 images

## Performance Metrics

| Metric | Value |
|--------|-------|
| mAP50 | 0.595 |
| mAP50-95 | 0.425 |
| Precision | 0.984 |
| Recall | 0.4 |
| Inference Speed | ~90ms per image |

## Key Technologies

- **YOLOv8**: Real-time object detection framework
- **PyTorch**: Deep learning framework
- **Flask**: Web framework for REST API
- **MLflow**: Experiment tracking and model registry
- **OpenCV**: Image processing
- **Docker**: Containerization

## How It Works
Road Image

↓

Flask API receives image

↓

Convert to numpy array (OpenCV)

↓

YOLOv8 model inference

↓

Draw bounding boxes on image

↓

Return:

Detection coordinates
Confidence scores
Annotated image

## Future Improvements

- [ ] Increase dataset size to improve recall
- [ ] Fine-tune hyperparameters (learning rate, epochs)
- [ ] Add data augmentation techniques
- [ ] Deploy to production with WSGI server
- [ ] Add real-time video stream detection
- [ ] Mobile app integration

## Learning Outcomes

This project demonstrates:
- Transfer learning with pre-trained models
- MLOps best practices (experiment tracking, versioning)
- REST API design and Flask
- Docker containerization
- GitHub version control
- Model evaluation metrics (precision, recall, mAP)

## Author

Mojgan Zahraee

## License

MIT License

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [MLflow Documentation](https://mlflow.org)
- [Flask Documentation](https://flask.palletsprojects.com)
- [Roboflow Dataset](https://universe.roboflow.com/rui-wei-vvdij/highway-guardrail-detection)