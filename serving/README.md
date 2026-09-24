# BentoML Model Serving Subsystem

Serves the trained PyTorch ResNet-50 visual defect detection model as a production microservice.

---

## Service Endpoints

- `POST /predict`: Quality inspection endpoint accepting image payloads and optional metadata.
  ```json
  {
    "prediction": "DEFECT",
    "confidence": 0.9421,
    "model_version": "v1.0.0",
    "latency_ms": 38.4,
    "probabilities": {
      "OK": 0.0579,
      "DEFECT": 0.9421
    }
  }
  ```
- `GET /health`: Liveness and health check endpoint.
- `GET /metrics`: Standard Prometheus metrics export.

---

## Running Locally

```bash
# Start BentoML service on port 3000
bentoml serve serving/service.py:svc --port 3000
```
