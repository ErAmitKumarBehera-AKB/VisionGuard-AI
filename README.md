# VisionGuard-AI: Visual Quality Inspection System for Manufacturing

An enterprise-grade visual quality inspection platform implementing the **TCS Industry-Aligned Capstone (Use Case B: Visual Quality Inspection System for Manufacturing)**.

Built with **PyTorch ResNet-50 transfer learning**, **real-time USB webcam & camera acquisition**, **BentoML model serving**, **FastAPI backend**, **React 18 / Vite single-page dashboard**, **Streamlit Human-in-the-Loop QC**, **Prometheus/Grafana telemetry**, **DVC data versioning**, and **Docker orchestration**.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Key Performance Benchmarks](#2-key-performance-benchmarks)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Dataset Pipeline & Preprocessing](#5-dataset-pipeline--preprocessing)
6. [Machine Learning Engineering](#6-machine-learning-engineering)
7. [Microservices Backend & Dual-Database Storage](#7-microservices-backend--dual-database-storage)
8. [Live Camera Acquisition & Real-Time Webcam Setup](#8-live-camera-acquisition--real-time-webcam-setup)
9. [Human-in-the-Loop QC & Web Dashboards](#9-human-in-the-loop-qc--web-dashboards)
10. [Observability & Industrial Telemetry](#10-observability--industrial-telemetry)
11. [Quickstart & Installation](#11-quickstart--installation)
12. [Docker Deployment](#12-docker-deployment)
13. [API Specification](#13-api-specification)
14. [Automated Testing & Quality Assurance](#14-automated-testing--quality-assurance)
15. [Repository Structure](#15-repository-structure)
16. [Jury Presentation & Capstone Compliance](#16-jury-presentation--capstone-compliance)

---

## 1. Executive Summary & Problem Statement

In discrete high-volume manufacturing (automotive powertrain machining, electronics PCB assembly, aerospace fasteners, and cast components), undetected surface defects lead to catastrophic field recalls, assembly line stoppages, and liability claims. 

### The Industry Challenge
* **Cognitive Fatigue**: Human inspector accuracy degrades by up to **30% after 45 minutes** of repetitive conveyor monitoring.
* **Subjective Inconsistency**: Shift-to-shift variance causes disparate scrap and rework rates.
* **Conveyor Speed Limits**: Modern lines operate at **2–5 parts per second**, far exceeding human perceptual limits.
* **Error Asymmetry**:
  * **False Negative (Defect Escape)**: Catastrophic cost (thousands to millions of dollars in recall damages).
  * **False Positive (False Alarm)**: Minor cost (a 5-second human verification of a quarantined part).

### The Solution: Zero-Defect Quality Priority
VisionGuard-AI enforces an **asymmetric loss penalty** (1.5x weight on defect misses) to prioritize **DEFECT Recall ($\ge 95\%$)** while maintaining **99.24% Precision**, sub-50ms inference, and a closed-loop Human-in-the-Loop retraining architecture.

---

## 2. Key Performance Benchmarks

Evaluated on an isolated test partition of **2,101 genuine industrial images** across 16 product categories:

| Performance Metric | Capstone Target | Achieved Result | Status | Industrial Impact |
|---|---|---|---|---|
| **Overall Accuracy** | $\ge 95.0\%$ | **98.71%** | **Exceeded** | Reliable, consistent line decisions |
| **Defect Recall (Sensitivity)** | $\ge 95.0\%$ | **97.87%** | **Exceeded** | **Only 20 defects missed out of 938 across 16 categories!** |
| **Defect Precision** | $\ge 90.0\%$ | **99.24%** | **Exceeded** | Near-zero false alarms (**only 7 false alarms out of 1,163 healthy parts**) |
| **Defect F1-Score** | $\ge 92.0\%$ | **98.55%** | **Exceeded** | Optimal balance between escape prevention and throughput |
| **ROC-AUC** | $\ge 98.0\%$ | **99.86%** | **Exceeded** | Exceptional discriminative power across varying defect types |
| **OK Specificity** | $\ge 95.0\%$ | **99.40%** | **Exceeded** | Prevents line clogging and operator fatigue |
| **Inference Latency (GPU)** | $< 100\text{ ms}$ | **34.2 ms** | **Exceeded** | High-throughput cloud serving |
| **Inference Latency (CPU / Webcam Loop)** | $< 150\text{ ms}$ | **88.5 ms** | **Exceeded** | Real-time interactive inspection on standard hardware |

### Confusion Matrix (Isolated Test Split: 2,101 Samples)
```
                  ┌───────────────────────┬───────────────────────┐
                  │ Predicted OK          │ Predicted DEFECT      │
┌─────────────────┼───────────────────────┼───────────────────────┤
│ Actual OK       │  TN = 1,156 (99.40%)  │  FP = 7     (0.60%)   │
├─────────────────┼───────────────────────┼───────────────────────┤
│ Actual DEFECT   │  FN = 20    (2.13%)   │  TP = 918   (97.87%)  │
└─────────────────┴───────────────────────┴───────────────────────┘
```

---

## 3. System Architecture

VisionGuard-AI operates on an asynchronous microservice architecture, separating real-time visual acquisition from deep learning inference, human verification, and telemetry governance.

```mermaid
graph TD
    subgraph Visual Acquisition Tier (Webcam / Industrial Camera)
        Webcam["USB Webcam / Industrial Camera (/dev/video0)"] --> FrameCapture["Frame Capture & ROI Focus Bounding Box"]
        FrameCapture --> Submit["Live Image Inspection Submission"]
    end

    subgraph Serving Tier
        Checkpoint["Model Checkpoint (best_model_384.pt)"] --> BentoML["BentoML Microservice (:3000)"]
    end

    subgraph Central Backend Tier
        Submit --> FastAPI["FastAPI Backend (:8000)"]
        BentoML <--> FastAPI
        FastAPI --> SQLite[("SQLite / PostgreSQL (Audit Trails)")]
        FastAPI --> Mongo[("MongoDB 7.0 (Users, Machines, Feedback)")]
    end

    subgraph User Experience & HITL Tier
        FastAPI --> WebApp["React 18 / Vite Web App with Live Camera (:8089 deployed, :5173 dev)"]
        FastAPI --> Streamlit["Streamlit Human-in-the-Loop QC (:8501)"]
        Streamlit --> OperatorAction["Operator Verification & Correction"]
        OperatorAction --> DVC["DVC Retraining Pool"]
        DVC --> Retrain["Automated Retraining Pipeline"]
    end

    subgraph Observability Tier
        FastAPI --> Prometheus["Prometheus (:9090)"]
        BentoML --> Prometheus
        Prometheus --> Grafana["Grafana Dashboards (:3001)"]
    end
```

---

## 4. Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Deep Learning** | PyTorch 2.2+, Torchvision | ResNet-50 backbone (`IMAGENET1K_V2`), custom classifier head, AMP mixed precision |
| **MLOps & Registry** | DVC, MLflow | Dataset cryptographic versioning, experiment tracking, model registry promotion |
| **Model Serving** | BentoML 1.2+ | Dynamic adaptive batching, async worker concurrency, Prometheus metrics |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 | High-concurrency REST API, OpenAPI docs, dependency injection, JWT authentication |
| **Databases** | MongoDB 7.0 & SQLite / PostgreSQL | Polyglot persistence: document-based identity/telemetry + relational audit trails |
| **Camera Acquisition** | USB Webcam, OpenCV, HTML5 MediaDevices | Real-time live camera capture, ROI focus framing, resolution control |
| **Web Frontend** | React 18, Vite, TypeScript, Tailwind CSS | Modern SPA: live camera stream, ROI focus framing, real-time KPI dashboards, Recharts |
| **Quality Control** | Streamlit, Pandas, Altair | Dedicated Human-in-the-Loop triage app for low-confidence (< 80%) inspection review |
| **Observability** | Prometheus v2.50.1 & Grafana 10.3.3 | Real-time line throughput, defect rate counters, latency percentiles (P50/P95/P99) |
| **Orchestration** | Docker & Docker Compose | Multi-container composition with healthchecks and network bridges |

---

## 5. Dataset Pipeline & Preprocessing

The training corpus unifies two industrial quality inspection datasets into a single stratified standard:

### A. Dataset Distribution (14,002 Total Images)
1. **MVTec Anomaly Detection (MVTec AD)**: 15 industrial product families:
   `cable`, `screw`, `metal_nut`, `transistor`, `bottle`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `pill`, `tile`, `toothbrush`, `wood`, `zipper`.
2. **Casting Product Image Data**: 8,648 images of cast submersible pump impellers captured under genuine foundry conditions (`ok_front` vs `def_front`).

### B. Stratified Leakage-Safe Splitting
* **Train Partition (70%)**: 9,800 images (backpropagation & parameter updates).
* **Validation Partition (15%)**: 2,101 images (early stopping & learning rate scheduling).
* **Test Partition (15%)**: 2,101 images (**strictly isolated**; zero cross-split leakage).
* **Class Balance**: 7,752 OK (55.4%) | 6,250 DEFECT (44.6%).

```bash
# Execute automated dataset ingestion & manifest creation
python ml/scripts/prepare_dataset.py --config ml/configs/dataset.yaml
```

---

## 6. Machine Learning Engineering

### Custom Classifier Architecture
The standard 1000-class ImageNet classification layer was removed and replaced with a specialized head:
$$\text{Input (2048)} \to \text{Dropout}(0.30) \to \text{Linear}(2048, 256) \to \text{BatchNorm1d}(256) \to \text{ReLU} \to \text{Dropout}(0.15) \to \text{Linear}(256, 2) \to [\text{OK}, \text{DEFECT}]$$

### Weighted Cross-Entropy Loss
$$\mathcal{L} = - \left( 1.0 \cdot y_{\text{OK}} \log p_{\text{OK}} + 1.5 \cdot y_{\text{DEFECT}} \log p_{\text{DEFECT}} \right) \quad \text{with label smoothing } \epsilon = 0.05$$
* Penalizes missed defects **1.5x heavier** than false alarms to guarantee high recall.
* Label smoothing prevents overconfidence on noisy industrial edge artifacts.

### Multi-Crop & Micro-Defect Fusion
Resizing $1024 \times 1024$ industrial images to $224 \times 224$ can destroy 2-pixel hairline scratches. 
`ml/scripts/evaluate_multicrop_fusion_384.py` extracts 5 native-resolution crops (four corners and center) fused with a global contextual view to catch sub-millimeter wire bends and surface pinholes.

```bash
# Train ResNet-50 defect detector
python ml/scripts/train.py --config ml/configs/training.yaml

# Evaluate checkpoint across all domain slices
python ml/scripts/evaluate.py --checkpoint ml/artifacts/checkpoints/best_model_384.pt
```

---

## 7. Microservices Backend & Dual-Database Storage

### Polyglot Persistence Architecture
* **Relational Storage (SQLite / PostgreSQL)**: Uses SQLAlchemy ORM for `inspections` and `feedbacks` tables, ensuring strict foreign-key integrity for ISO 9001 compliance.
* **Document Storage (MongoDB 7.0)**: Manages `users`, `machines`, `audit_logs`, and the dynamic `feedback` retraining queue.

### Deterministic 80% Decision Gating (`decision_service.py`)
```python
if confidence < 0.80:
    return Decision("PENDING_REVIEW", "PENDING_HUMAN_REVIEW", review_required=True)
elif prediction == "OK":
    return Decision("COMPLETED", "AUTOMATIC_OK", review_required=False, final_label="OK")
else:
    return Decision("COMPLETED", "AUTOMATIC_DEFECT", review_required=False, final_label="DEFECT")
```

---

## 8. Live Camera Acquisition & Real-Time Webcam Setup

The system integrates real-time camera acquisition using standard USB webcams, industrial UVC cameras, or integrated laptop cameras for flexible plant and laboratory deployment:

| Acquisition Mode | Technology | Capabilities | Use Case |
|---|---|---|---|
| **Web-Based Live Camera** | HTML5 MediaDevices / WebRTC | Real-time viewport, interactive ROI focus box, digital zoom, mirror mode | Interactive operator workstation (`:8089/inspect`) |
| **Direct OpenCV Camera** | OpenCV (`cv2.VideoCapture`) | Direct frame grabbing from `/dev/video0`, configurable resolution (640x480 up to 4K) | Automated script runner & headless capture (`edge/scripts/camera_interface.py`) |
| **Diagnostic Fallback** | PIL Synthetic Generator | Mock frame generator with simulated OK/DEFECT cycles | Automated testing and CI/CD validation without physical camera hardware |

### Interactive ROI (Region of Interest) Focus Framing
In factory environments, background clutter (conveyor belt texture, nearby machinery, ambient room lighting) can introduce noise.
The React Web Dashboard provides an **interactive focus frame**:
1. Operators position the target component inside the centered inspection focus guide.
2. The UI extracts the cropped bounding box, eliminating background distractions.
3. The focused crop is transmitted directly to the inference service (`/api/v1/admin/inspect` or `/api/v1/inspection/predict`) for sub-50ms scoring.

```bash
# Run camera capture and automated inspection loop via connected webcam
python edge/scripts/edge_runner.py --cycles 10
```

---

## 9. Human-in-the-Loop QC & Web Dashboards

### A. Streamlit Quality Control Dashboard (`:8501`)
* Accessible at `http://localhost:8501`.
* **Low-Confidence Triage Queue**: Filters items with confidence $< 80\%$ or flagged defects.
* **Side-by-Side Review**: Operators inspect photographic evidence and confirm or correct labels.
* **Retraining Pipeline Integration**: Approved corrections enter the DVC retraining pool.

### B. React 18 / Vite VisionInspect AI Web App (`:8089` deployed, `:5173` development)
* Docker deployment: `http://localhost:8089`.
* Local Vite development: `http://localhost:5173`.
* **Live Camera Interface**: Connects to webcams, industrial cameras, or Insta360 4K sensors with zoom, mirror, and ROI bounding box framing.
* **Operational KPI Dashboard**: Real-time pass rates, defect rates, average latency, and line statistics.
* **Machine Administration**: Configure multiple conveyor lines and view model registry statuses.

---

## 10. Observability & Industrial Telemetry

Prometheus scrapes backend and BentoML metrics every 5 seconds, rendered in pre-configured Grafana dashboards:

* **Prometheus UI**: `http://localhost:9090`
* **Grafana Dashboard**: `http://localhost:3001` (Credentials: `admin` / `admin`)
  * Throughput (Parts per second)
  * Real-time Defect Rate (%)
  * Latency Percentiles ($P_{50}$, $P_{95}$, $P_{99}$)
  * Epistemic Uncertainty & Low-Confidence Alerts

---

## 11. Quickstart & Installation

### Prerequisites
* Linux (Ubuntu 22.04+ / Debian 12), macOS, or Windows WSL2
* Python 3.11+
* Node.js 18+ and pnpm (for frontend development)
* Docker & Docker Compose
* Standard USB Webcam or integrated camera (for live inspection)

### Step 1: Clone Repository
```bash
git clone https://github.com/ErAmitKumarBehera-AKB/VisionGuard-AI.git
cd VisionGuard-AI
```

### Step 2: Environment Configuration
```bash
# Copy and verify environment variables
cp .env.example .env
```

### Step 3: Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Run Services Locally
```bash
# Terminal 1: BentoML Model Serving (:3000)
bentoml serve serving/service.py:svc --port 3000

# Terminal 2: FastAPI Backend (:8000)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 3: Streamlit QC Interface (:8501)
streamlit run qc/app.py --server.port 8501

# Terminal 4: React Web Application (:5173)
cd frontend
pnpm install
pnpm --filter @workspace/visioninspect-ai run dev
```

---

## 12. Docker Deployment

Deploy the entire production stack with a single command:

```bash
# Build and launch all services in detached mode
docker compose up -d --build

# Inspect service logs
docker compose logs -f

# Check container health status
docker compose ps

# Teardown ecosystem
docker compose down
```

### Deployed Services Port Mapping
* **Web Application**: `http://localhost:8089`
* **FastAPI Backend (Swagger Docs)**: `http://localhost:8000/docs`
* **Streamlit QC Interface**: `http://localhost:8501`
* **BentoML Model Serving**: `http://localhost:3000`
* **Grafana Telemetry Dashboard**: `http://localhost:3001`
* **Prometheus Time-Series DB**: `http://localhost:9090`
* **MongoDB**: `localhost:27017`

---

## 13. API Specification

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/v1/inspection/predict` | Submit part image for automated classification | Optional |
| `GET` | `/api/v1/inspection/history` | Paginated inspection history with category/result filters | Bearer |
| `GET` | `/api/v1/inspection/{id}` | Retrieve specific inspection record and photographic evidence | Bearer |
| `GET` | `/api/v1/inspection/stats/summary`| Aggregate statistics (defect rate, average latency, total count)| Bearer |
| `POST` | `/api/v1/feedback` | Record operator confirmation or label correction | Bearer |
| `GET` | `/api/v1/feedback/export` | Export validated feedback for DVC dataset ingestion | Bearer |
| `POST` | `/api/v1/auth/login` | Authenticate user; returns signed JWT token | Public |
| `POST` | `/api/v1/admin/inspect` | Web-based live camera inspection with quarantine logic | Bearer |
| `POST` | `/api/v1/retraining/trigger` | Trigger governed model retraining on approved feedback | Admin |
| `GET` | `/health` | Liveness and readiness probe | Public |
| `GET` | `/metrics` | Prometheus metrics scrape endpoint | Public |

---

## 14. Automated Testing & Quality Assurance

Automated unit, integration, and security tests validate both the ML pipeline and Backend APIs:

```bash
# Run complete test suite across ML and Backend
pytest backend/tests ml/tests -v
```

### Test Coverage Highlights
* `backend/tests/test_decision_service.py`: Validates deterministic 80% confidence gating.
* `backend/tests/test_api.py`: Tests prediction flow, feedback loops, and metrics endpoints.
* `backend/tests/test_auth_security.py`: Validates Bcrypt hashing, password complexity, and JWT tampering prevention.
* `ml/tests/test_inference.py`: Verifies tensor transformation pipelines and batch scoring.
* `ml/tests/test_metrics.py`: Confirms precision, recall, and confusion matrix arithmetic.

---

## 15. Repository Structure

```
VisionGuard-AI/
├── backend/                  # FastAPI Backend API
│   ├── app/
│   │   ├── api/v1/           # REST endpoints (inspection, feedback, system, secure)
│   │   ├── auth/             # JWT tokens, bcrypt security, RBAC
│   │   ├── models/           # SQLAlchemy ORM models (inspections, feedback)
│   │   ├── schemas/          # Pydantic v2 validation schemas
│   │   ├── services/         # Inspection service, decision engine, BentoML client
│   │   ├── utils/            # MongoDB & SQLite connection managers
│   │   ├── config.py         # App configuration & environment loader
│   │   └── main.py           # FastAPI ASGI entrypoint
│   ├── tests/                # Pytest integration & security test suite
│   └── Dockerfile            # Multi-stage production Docker build
├── edge/                     # Camera interface, live capture scripts, and simulation runner
│   ├── config/               # Edge & camera YAML configuration
│   ├── scripts/              # Camera HAL, OpenCV capture, and test runner
│   ├── Dockerfile.arm64      # ARM64 container definition
│   └── docker-compose.arm64.yml
├── frontend/                 # React 18 / Vite / TypeScript Dashboard
│   ├── artifacts/visioninspect-ai/ # React SPA source code
│   ├── nginx.conf            # Reverse proxy configuration
│   └── Dockerfile            # Nginx production container build
├── ml/                       # Machine Learning Engineering Core
│   ├── artifacts/checkpoints/# Trained ResNet-50 weights & metadata (best_model_384.pt)
│   ├── configs/              # Dataset, model, and training YAML configs
│   ├── data/manifests/       # Unified dataset manifests (CSV / Parquet)
│   ├── scripts/              # Data prep, training, multi-crop evaluation, analysis
│   ├── src/                  # Loaders, ResNet-50 model, predictor, trainer
│   └── tests/                # Dataset & model unit tests
├── monitoring/               # Prometheus & Grafana Configuration
│   ├── grafana/dashboards/   # Pre-provisioned industrial inspection dashboards
│   └── prometheus/           # Prometheus scraping configuration
├── qc/                       # Streamlit Human-in-the-Loop QC Portal
│   ├── components/           # Inspection review cards, sidebar, metrics
│   └── app.py                # Streamlit entrypoint
├── docker-compose.yml        # Multi-service container orchestration
├── dvc.yaml                  # Reproducible DVC pipeline definition
├── Makefile                  # Automation convenience targets
└── README.md                 # Primary system documentation
```

---

## 16. Jury Presentation & Capstone Compliance

### Capstone Track Compliance
This project strictly fulfills all requirements of **TCS Industry-Aligned Capstone (Use Case B: Visual Quality Inspection System for Manufacturing)**:
1. **Industry-Relevant Problem**: Zero-defect manufacturing quality control with asymmetric error weighting.
2. **Deep Learning Core**: PyTorch ResNet-50 transfer learning exceeding the 95% defect recall mandate (**97.87% achieved**).
3. **Multi-Domain Ingestion**: Evaluated across 16 industrial categories (MVTec AD + Real Foundry Casting data).
4. **Real-Time Visual Acquisition**: Live USB webcam & industrial camera capture, interactive ROI focus framing, and sub-50ms inference.
5. **Human-in-the-Loop Governance**: Streamlit QC triage for low-confidence (< 80%) cases with DVC feedback versioning.
6. **Production MLOps**: Containerized microservices, BentoML serving, and Prometheus/Grafana observability.

### Complete Jury Defense Manual
For comprehensive technical defense preparation—including **20 tough jury questions & answers**, **camera optical calculations & threshold derivations**, and a **minute-by-minute speaking script**—refer to:
* **Word Document Guide**: [`Visual_Quality_Inspection_System_Jury_Defense_Guide.docx`](Visual_Quality_Inspection_System_Jury_Defense_Guide.docx)
* **Plain Text Guide**: [`Visual_Quality_Inspection_System_Jury_Defense_Guide.txt`](Visual_Quality_Inspection_System_Jury_Defense_Guide.txt)

---

## 🔒 Confidentiality & Project Status
This repository is a **private, proprietary capstone project** developed for the **TCS Industry-Aligned Capstone (Use Case B: Visual Quality Inspection System for Manufacturing)**. All rights reserved. Unauthorized copying, distribution, or commercial use is strictly prohibited.
