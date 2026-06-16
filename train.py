import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import yaml
import os

# ── 1. Setup MLflow ──
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment("guardrail-detection")

# ── 2. Load data config ──
with open("dataset/data.yaml", "r") as f:
    data_config = yaml.safe_load(f)

print("Dataset Config:")
print(f"Classes: {data_config['names']}")
print(f"Number of classes: {data_config["nc"]}")

# ── 3. Train with MLflow tracking ──

with mlflow.start_run():
    
    # Log parameters
    params = {
        "model" : "yolov8s.pt",
        "epochs" : 20,
        "imgsz" :  640,
        "batch" : 8,
        "lr0" : 0.01,
        "dataset" : "highway-guardrail-detection"
    }
    mlflow.log_params(params)
    
    # Load YOLOv8 model
    model = YOLO("yolov8s.pt")
    
    # Train the model
    results = model.train(
        data = "dataset/data.yaml",
        epochs = params['epochs'],
        imgsz = params['imgsz'],
        batch = params['batch'],
        lr0 = params['lr0'],
        name = 'guardrail_detector',
        exist_ok = True
         
    )
    
    # Log metrics
    mlflow.log_metric('mAP50', results.results_dict.get('metrics/mAP50(B)', 0))
    mlflow.log_metric('mAP50-95', results.results_dict.get('metrics/mAP50-95(B)', 0))
    mlflow.log_metric('precision', results.results_dict.get('metrics/precision(B)', 0))
    mlflow.log_metric("recall", results.results_dict.get('metrics/recall(B)' , 0))
    
    # Save model to MLflow
    mlflow.pytorch.log_model(model.model, "guardrail_model")
    print('\nTraining compelet!')
    (f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    
