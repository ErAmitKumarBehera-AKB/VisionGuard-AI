# API Specification & Replit Frontend Integration Guide

This guide details how external frontends (such as your future Replit-generated frontend) interact with the backend API.

---

## Base URL

- Development: `http://localhost:8000`
- Production / Edge Gateway: Configurable via `.env` (`BACKEND_HOST`, `BACKEND_PORT`)

---

## CORS Configuration

The backend is configured with FastAPI `CORSMiddleware` supporting:
- Origins: `*`, `http://localhost:3000`, `http://localhost:5173`, `http://localhost:8089`, Replit subdomains
- Methods: `GET`, `POST`, `OPTIONS`
- Headers: `*`

---

## Key Endpoints for the Frontend

### 1. Submit Inspection (`POST /api/v1/inspection/predict`)

- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `image`: Image file (PNG/JPEG)
  - `product_category`: (Optional string, e.g., `'screw'`, `'cable'`, `'metal_nut'`)
  - `source_dataset`: (Optional string, e.g., `'mvtec'`, `'casting'`)
- **Response** (`200 OK`):
  ```json
  {
    "inspection_id": "INSP-A1B2C3D4E5F6",
    "prediction": "DEFECT",
    "confidence": 0.9412,
    "model_version": "v1.0.0",
    "latency_ms": 34.2,
    "product_category": "metal_nut",
    "probabilities": {
      "OK": 0.0588,
      "DEFECT": 0.9412
    },
    "timestamp": "2026-09-23T16:00:00Z",
    "is_low_confidence": false
  }
  ```

### 2. Inspection History (`GET /api/v1/inspection/history`)

- **Query Parameters**:
  - `page`: Integer (default 1)
  - `page_size`: Integer (default 20)
  - `prediction`: `'OK'` or `'DEFECT'` (optional)
  - `low_confidence_only`: Boolean (default `false`)
- **Response** (`200 OK`):
  ```json
  {
    "total": 1420,
    "page": 1,
    "page_size": 20,
    "items": [
      {
        "inspection_id": "INSP-A1B2C3D4E5F6",
        "timestamp": "2026-09-23T16:00:00Z",
        "prediction": "DEFECT",
        "confidence": 0.9412,
        "model_version": "v1.0.0",
        "latency_ms": 34.2,
        "product_category": "metal_nut",
        "operator_label": null,
        "feedback_status": "PENDING"
      }
    ]
  }
  ```

### 3. Summary Statistics (`GET /api/v1/inspection/stats/summary`)

- **Response** (`200 OK`):
  ```json
  {
    "total_inspections": 1420,
    "defect_count": 86,
    "ok_count": 1334,
    "defect_rate_percentage": 6.06,
    "average_latency_ms": 32.4,
    "low_confidence_count": 12,
    "reviewed_count": 45
  }
  ```

### 4. Submit Human Feedback (`POST /api/v1/feedback`)

- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "inspection_id": "INSP-A1B2C3D4E5F6",
    "operator_label": "DEFECT",
    "comments": "Confirmed bent wire on cable terminal."
  }
  ```
- **Response** (`201 Created`):
  ```json
  {
    "feedback_id": 14,
    "inspection_id": "INSP-A1B2C3D4E5F6",
    "model_prediction": "DEFECT",
    "operator_label": "DEFECT",
    "confidence": 0.9412,
    "model_version": "v1.0.0",
    "comments": "Confirmed bent wire on cable terminal.",
    "created_at": "2026-09-23T16:05:00Z",
    "status": "RECORDED"
  }
  ```
