# Visual Quality Inspection System for Manufacturing

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![BentoML](https://img.shields.io/badge/BentoML-1.2+-black.svg)](https://www.bentoml.com/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945dd6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, industry-aligned visual quality inspection platform implementing **TCS Industry-Aligned Capstone (Use Case B: Visual Quality Inspection System for Manufacturing)**.

Built with **PyTorch ResNet-50 transfer learning**, **DVC dataset versioning**, **MLflow tracking and model registry**, **BentoML model serving**, **FastAPI backend**, **Streamlit Human-in-the-Loop Quality Control**, **Prometheus/Grafana observability**, **Local PC x86_64 edge integration**, and **Docker orchestration**.

---

## 1. Project Objective

In modern manufacturing (automotive, electronics, precision engineering), shipping a defective component (False Negative) leads to catastrophic recalls, warranty claims, and safety liabilities. Conversely, excessive false alarms (False Positives) waste operator labor.

This project delivers:
1. **Zero-Defect Quality Priority**: A deep transfer learning vision pipeline tuned to prioritize **DEFECT Recall (>= 95%)** using weighted loss and threshold optimization.
2. **Domain-Aware Generalization**: Ingests and validates multi-domain industrial datasets (**MVTec AD** and **Casting Product Image Data**) to ensure models do not merely memorize background textures.
3. **Edge-to-Core Continuum**: Deployable on **Local PC** for real-time line inference, paired with a central microservice architecture for auditing and human-in-the-loop validation.
4. **Continuous Feedback Retraining Loop**: Captures operator corrections, versions datasets with DVC, and triggers governed retraining pipelines.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Edge["Conveyor Line (Edge Node)"]
        direction TB
        S["Photoelectric Sensor Trigger (GPIO 24)"]
        C["Webcam"]
        R["Edge Runner (ResNet-50 / BentoML Client)"]
        A["Pneumatic Rejection Actuator (GPIO 18)"]
        
        S --> R
        C --> R
        R --> A
    end

    subgraph Cloud["Central Platform (Local Docker / On-Premise)"]
        direction TB
        B["BentoML Model Serving (:3000)"]
        F["FastAPI Backend (:8000)"]
        DB[("Audit Database (SQLite/PostgreSQL)")]
        QC["Streamlit QC Interface (:8501)"]
        UI["React Frontend Dashboard (:5173)"]
        M["Prometheus & Grafana (:9090 / :3001)"]
        
        F <--> B
        F --> DB
        QC <--> F
        UI <--> F
        B -.-> M
        F -.-> M
    end

    R -- "Async Telemetry / API Requests" --> F
```

---

## 3. Dataset Setup & Exact Placement

Because the datasets total over 6GB, they are not stored on GitHub. Your teammates must manually download them from their sources (like Kaggle) and extract them into the correct folders before running the system.

### A. MVTec Anomaly Detection (MVTec AD)
**Step 1:** Download the dataset from the [MVTec AD Dataset Portal](https://www.mvtec.com/company/research/datasets/mvtec-ad).
**Step 2:** Extract the downloaded ZIP file.
**Step 3:** Move the extracted category folders (`cable`, `screw`, `metal_nut`, `transistor`) exactly into the `ml/data/raw/mvtec/` directory.

- **Target Categories**: `cable`, `screw`, `metal_nut`, `transistor`
- **Expected Directory Structure**:
  ```
  ml/data/raw/mvtec/
  ├── cable/
  │   ├── train/good/*.png
  │   └── test/{good, bent_wire, cable_swap, cut_inner_insulation, ...}/*.png
  ├── screw/
  │   ├── train/good/*.png
  │   └── test/{good, manipulated_front, scratch_head, thread_side, ...}/*.png
  ├── metal_nut/
  │   ├── train/good/*.png
  │   └── test/{good, bent, color, flip, scratch}/*.png
  └── transistor/
      ├── train/good/*.png
      └── test/{good, bent_lead, cut_lead, damaged_case, misplaced}/*.png
  ```

### B. Casting Product Image Data for Quality Inspection
**Step 1:** Download the dataset from [Kaggle - Casting Product Image Data](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).
**Step 2:** Extract the downloaded ZIP file (specifically the `casting_data/casting_data/` folder).
**Step 3:** Move the `ok_front` and `def_front` folders exactly into the `ml/data/raw/casting/` directory.

- **Expected Directory Structure**:
  ```
  ml/data/raw/casting/
  ├── ok_front/*.jpeg
  └── def_front/*.jpeg
  ```

---

## 4. Dataset Licensing Considerations

- **MVTec AD**: Distributed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Appropriate for academic, educational, and research evaluation.
- **Casting Product Image Data**: Distributed under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
- **Data Governance**: Large raw datasets must never be checked into Git. Use DVC to version dataset manifests and checksums.

---

## 5. Environment Setup

### Prerequisites
- Linux / macOS (x86_64 or x86_64) or Windows WSL2
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or standard pip
- Docker & Docker Compose (optional, for containerized run)

### Commands
```bash
# 1. Clone repository
cd TCS_project

# 2. Setup virtual environment
# Option A: Using uv (Fast)
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Option B: Using automated setup script
bash scripts/setup.sh

# 3. Verify PyTorch and CUDA installation
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```

---

## 6. Dataset Preparation Pipeline

The ingestion pipeline scans the raw dataset directories, validates file integrity, normalizes labels into binary target classes (`0 = OK`, `1 = DEFECT`), preserves product and domain metadata, and creates stratified, leakage-safe splits.

```bash
# Execute dataset preparation
python ml/scripts/prepare_dataset.py --config ml/configs/dataset.yaml

# Generated Outputs:
#   ml/data/manifests/unified_manifest.csv
#   ml/data/manifests/unified_manifest.parquet
```

---

## 7. DVC (Data Version Control) Workflow

Version manifests and dataset pipelines with DVC:

```bash
# 1. Initialize DVC repository
dvc init

# 2. Configure remote storage (e.g., local storage or S3/MinIO)
dvc remote add -d local_storage /tmp/dvc_storage

# 3. Reproduce the full DVC pipeline (prepare -> train -> evaluate)
dvc repro

# 4. View pipeline status and dependency DAG
dvc dag
```

---

## 8. Model Training

Fine-tunes **ResNet-50** with weighted loss, Cosine Annealing scheduler, and automatic mixed precision (AMP) on CUDA or CPU:

```bash
# Run training with default configurations
python ml/scripts/train.py --config ml/configs/training.yaml

# Or using the convenience script:
bash scripts/train.sh --epochs 15 --batch-size 32 --lr 0.0001
```

Checkpoints and training metadata are saved to `ml/artifacts/checkpoints/best_model.pt`.

---

## 9. Model Evaluation & Domain-Aware Analysis

Evaluates performance on the isolated test partition across overall, per-domain, and per-product slices:

```bash
python ml/scripts/evaluate.py --checkpoint ml/artifacts/checkpoints/best_model.pt

# Reports generated:
#   ml/artifacts/reports/evaluation_metrics.json
#   ml/artifacts/reports/confusion_matrix.png
```

---

## 10. MLflow Experiment Tracking & Model Registry

Launch the local MLflow tracking server:

```bash
# Start MLflow UI
mlflow ui --port 5000
```
Open `http://localhost:5000` to inspect:
- Training and validation loss curves.
- Defect recall, precision, and F1 per epoch.
- Model artifacts and confusion matrix images.
- Model Registry: `ManufacturingDefectResNet50`.

---

## 11. BentoML Model Serving

Serve the trained model with high-throughput batching and Prometheus metrics:

```bash
# Serve model on port 3000
bentoml serve serving/service.py:svc --port 3000
```

Test inference:
```bash
curl -X POST http://localhost:3000/predict \
     -H "Content-Type: multipart/form-data" \
     -F "image=@ml/data/raw/mvtec/metal_nut/test/scratch/scratch_001.png"
```

---

## 12. FastAPI Backend API

Run the central backend API:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- Prometheus Metrics: `http://localhost:8000/metrics`

---

## 13. Streamlit Human-in-the-Loop QC Interface

Launch the operator dashboard:

```bash
streamlit run qc/app.py --server.port 8501
```
Open `http://localhost:8501` to:
- Review live inspections and triage low-confidence parts (`< 80%`).
- Inspect component images and verify / correct predicted labels.
- Submit feedback into the retraining pool.

---

## 14. Monitoring (Prometheus & Grafana)

Prometheus scrapes metrics from the backend and BentoML serving endpoints every 5 seconds.

- **Prometheus UI**: `http://localhost:9090`
- **Grafana Dashboard**: `http://localhost:3001` (Credentials: `admin` / `admin`)
  - Dashboards pre-provisioned in `monitoring/grafana/dashboards/inspection_dashboard.json`.

---

## 15. Docker Deployment

Launch the entire ecosystem with Docker Compose:

```bash
# 1. Create your environment file
cp .env.example .env

# 2. Build and run backend, serving, qc, frontend, prometheus, and grafana
docker compose up -d --build

# View logs
docker compose logs -f

# Teardown
docker compose down
```


---

## 16. Edge Deployment (Local PC)

Targeting an 8GB Local PC with hardware sensor stubs:

```bash
# Run edge runner in development mode
python edge/scripts/edge_runner.py --cycles 10

# Or build and launch x86_64 container on Local PC:
docker compose -f edge/docker-compose.edge.yml up -d --build
```

---

## 17. Human-in-the-Loop & Continuous Retraining Workflow

```mermaid
flowchart TD
    A["1. Conveyor / Client Image Inspection"] --> B["2. Prediction: DEFECT or OK (with Confidence Score)"]
    B --> C{"3. Confidence < 80% or Operator Flagged?"}
    
    C -- "YES" --> D["Route to Streamlit QC App (:8501)"]
    D --> E["Human Operator Corrects / Confirms Label"]
    E --> F["Appended to Retraining Feedback Pool"]
    F --> G["DVC Dataset Version Increment (dvc commit)"]
    G --> H["Trigger Retraining (scripts/train.sh)"]
    H --> I["Model Evaluation & Safety Gate (Recall >= 95%)"]
    I --> J["MLflow Model Registry Promotion"]
    
    C -- "NO" --> K["Automated Actuator Decision & Line Output"]
```

---

## 18. Integrated React Frontend Dashboard

The `frontend/` directory contains a full modern React application built with Vite and Tailwind CSS. 
It is automatically orchestrated by Docker Compose on port `5173`. 
The dashboard provides a real-time command center for monitoring inspections, metrics, and quality alerts.

---

## 19. How Replit Frontend Communicates with the Backend API

The FastAPI backend has CORS enabled (`allow_origins=["*"]`) and provides the following REST interfaces for the frontend:

1. **Submit Inspection**: `POST http://localhost:8000/api/v1/inspection/predict` (Multipart FormData with `image`, `product_category`).
2. **Inspection History**: `GET http://localhost:8000/api/v1/inspection/history?page=1&page_size=20`.
3. **Inspection Details**: `GET http://localhost:8000/api/v1/inspection/{inspection_id}`.
4. **Summary Metrics**: `GET http://localhost:8000/api/v1/inspection/stats/summary`.
5. **Submit Feedback**: `POST http://localhost:8000/api/v1/feedback`.
6. **System Status**: `GET http://localhost:8000/api/v1/system/status`.

---

## 20. Running Automated Tests

```bash
# Run all unit and integration tests across ML and Backend
pytest ml/tests backend/tests -v
```
