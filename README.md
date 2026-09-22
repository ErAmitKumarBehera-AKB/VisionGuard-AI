<div align="center">
  
# 👁️ VisionGuard-AI
**TCS Xcelerate Capstone Project | Visual Quality Inspection System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=flat&logo=mlflow&logoColor=blue)](https://mlflow.org/)
[![Grafana](https://img.shields.io/badge/grafana-%23F46800.svg?style=flat&logo=grafana&logoColor=white)](https://grafana.com/)

An end-to-end MLOps pipeline for automated, high-accuracy Visual Quality Inspection. Designed to detect manufacturing defects in real-time using Computer Vision.

</div>

---

## 🚀 Overview

VisionGuard-AI automates the visual inspection process on manufacturing lines. By leveraging deep learning, it accurately classifies products as defective or non-defective. The system is built with a robust **MLOps architecture**, ensuring seamless model training, serving, and production monitoring.

### ✨ Key Features
- **Deep Learning Vision Model:** Powered by PyTorch for high-accuracy defect detection.
- **Experiment Tracking:** Integrated with MLflow to track model parameters, metrics, and artifacts (`mlruns/`).
- **Real-Time Monitoring:** Prometheus and Grafana dashboards for live system health and prediction monitoring.
- **Containerized Deployment:** Fully Dockerized for "one-click" reproducible environments.
- **Model Serving:** Production-ready inference API.

---

## 📂 Project Structure

```bash
VisionGuard-AI/
├── src/
│   ├── app.py              # Main application/API for inference
│   ├── train.py            # Model training script
│   ├── serve.py            # Model serving logic
│   └── generate_data.py    # Data simulation/generation script
├── models/
│   └── model.pth           # Trained PyTorch model weights
├── mlruns/                 # MLflow experiment tracking logs
├── grafana/                # Grafana dashboard configurations
├── prometheus.yml          # Prometheus monitoring configuration
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Getting Started (For Teammates)

Follow these instructions to set up the project on your local machine and start contributing!

### 1. Prerequisites
Ensure you have the following installed on your system:
* [Python 3.10+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Optional, but recommended for running the full stack)

### 2. Clone the Repository
```bash
git clone https://github.com/ErAmitKumarBehera-AKB/VisionGuard-AI.git
cd VisionGuard-AI
```

### 3. Set Up a Virtual Environment (Highly Recommended)
Creating a virtual environment ensures that your dependencies don't conflict with other projects.
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Mac/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Project

### Option A: Running Locally (Python)
To run the model serving API directly on your machine:
```bash
python src/app.py
```

### Option B: Running the Full Stack (Docker)
To spin up the App, Prometheus, and Grafana all at once:
```bash
docker-compose up --build
```
* **App API:** `http://localhost:5000` (or the port defined in app.py)
* **Grafana Dashboard:** `http://localhost:3000`
* **Prometheus:** `http://localhost:9090`

---

## 🤝 How to Contribute

We are building this together to win the hackathon! 🏆 Here is the workflow:

1. **Pull the latest changes:** Always start by getting the newest code from the main branch.
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Create a new branch:** Name it based on what you are working on.
   ```bash
   git checkout -b feature-add-new-model
   ```
3. **Make your changes:** Write code, train models, or fix bugs!
4. **Commit and Push:**
   ```bash
   git add .
   git commit -m "Brief description of what you did"
   git push origin feature-add-new-model
   ```
5. **Create a Pull Request:** Go to GitHub and open a Pull Request so the team can review your code before it merges into `main`.

---
*Built for the TCS Xcelerate Capstone Hackathon 2026*
