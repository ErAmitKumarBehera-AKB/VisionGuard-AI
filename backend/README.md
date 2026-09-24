# Backend API Subsystem - FastAPI

The Backend API coordinates visual defect inspection requests, interfaces with the BentoML model serving microservice, persists inspection audit records, records human-in-the-loop operator feedback, and exposes Prometheus metrics.

> [!IMPORTANT]
> The backend operates strictly independently of the frontend. All endpoints support standard REST semantics and CORS, ready for the upcoming Replit-generated frontend.

---

## REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness / readiness probe |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint |
| `POST` | `/api/v1/inspection/predict` | Multipart image inspection endpoint |
| `GET` | `/api/v1/inspection/history` | Paginated inspection audit history |
| `GET` | `/api/v1/inspection/{inspection_id}` | Detailed inspection record |
| `GET` | `/api/v1/inspection/stats/summary` | Summary KPIs (defect rate, throughput, etc.) |
| `POST` | `/api/v1/feedback` | Human operator label confirmation or correction |
| `GET` | `/api/v1/feedback/export` | Export validated feedback for DVC retraining |
| `GET` | `/api/v1/system/status` | Comprehensive hardware, BentoML, and DB status |

---

## Interactive API Documentation

When the backend is running, explore and test the OpenAPI schemas interactively:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Running Locally

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
