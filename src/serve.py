import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import bentoml
from prometheus_client import Counter, Histogram

# Declare Prometheus metrics (BentoML exposes these automatically under /metrics)
INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total number of casting part inference requests",
    labelnames=["class_prediction"]
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence score distribution of predictions",
    labelnames=["class_prediction"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
)

@bentoml.service(
    name="casting_inspection",
    traffic={"timeout": 60}
)
class CastingInspectionService:
    def __init__(self):
        # Determine device (CPU preferred for local edge simulation)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BentoML service initializing on device: {self.device}")
        
        # Recreate ResNet-50 architecture
        try:
            self.model = models.resnet50(weights=None)
        except AttributeError:
            self.model = models.resnet50()
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # Load weights
        model_path = os.path.join("models", "model.pth")
        if os.path.exists(model_path):
            try:
                payload = torch.load(model_path, map_location=self.device)
                if isinstance(payload, dict) and "model_state_dict" in payload:
                    self.model.load_state_dict(payload["model_state_dict"])
                else:
                    self.model.load_state_dict(payload)
                print(f"Successfully loaded trained weights from {model_path}")
            except Exception as e:
                print(f"Error loading weights: {e}. Running with uninitialized weights.")
        else:
            print(f"WARNING: Weights not found at {model_path}. Running with random initialization.")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class mapping: ImageFolder lists classes alphabetically: ['defective', 'ok']
        self.classes = ["defective", "ok"]

    @bentoml.api
    def predict(self, img: Image.Image) -> dict:
        start_time = time.time()
        
        # Preprocess image
        img_tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)
            
        # Extract results
        pred_class = self.classes[pred_idx.item()]
        conf_score = confidence.item()
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log to Prometheus
        INFERENCE_REQUESTS.labels(class_prediction=pred_class).inc()
        INFERENCE_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.labels(class_prediction=pred_class).observe(conf_score)
        
        print(f"Prediction: {pred_class} | Confidence: {conf_score:.4f} | Latency: {latency:.4f}s")
        
        return {
            "prediction": pred_class,
            "confidence": float(conf_score),
            "latency_seconds": float(latency),
            "status": "success"
        }
