# System Architecture Specification

## Overview

The **Visual Quality Inspection System for Manufacturing** is an end-to-end industrial platform designed to detect visual manufacturing defects with high precision and near-zero false-negative tolerance.

```mermaid
graph TD
    subgraph Data Tier
        MVTec["MVTec AD Dataset"] --> Loader["Unified Dataset Loader"]
        Casting["Casting Dataset"] --> Loader
        Loader --> Manifest["Unified Manifest (Parquet/CSV)"]
        Manifest --> DVC["DVC Version Control"]
        Manifest --> Split["Stratified Leakage-Safe Splitter"]
    end

    subgraph ML Pipeline
        Split --> Train["train.py (PyTorch ResNet-50)"]
        Train --> Eval["evaluate.py (Domain-Aware)"]
        Train --> MLflow["MLflow Tracking & Model Registry"]
        MLflow --> Checkpoint["Best Checkpoint (best_model.pt)"]
    end

    subgraph Edge Tier (Raspberry Pi 5)
        Conveyor["Conveyor Line"] --> Sensor["Photoelectric Sensor (Pin 24)"]
        Sensor --> Cam["Industrial Camera (/dev/video0)"]
        Cam --> EdgeNode["Edge Runner (ResNet-50)"]
        EdgeNode --> Actuator["Pneumatic Ejector (Pin 18)"]
        EdgeNode --> Telemetry["Async Telemetry Sync"]
    end

    subgraph Serving Tier
        Checkpoint --> Bento["BentoML Microservice (:3000)"]
        Telemetry --> Backend["FastAPI Backend (:8000)"]
        Bento --> Backend
        Backend --> DB[(SQLite / PostgreSQL Audit DB)]
    end

    subgraph Human & Analytics Tier
        Backend --> Streamlit["Streamlit HITL QC App (:8501)"]
        Streamlit --> HumanFeedback["Operator Label Correction"]
        HumanFeedback --> RetrainPool["Retraining Data Pool (DVC)"]
        RetrainPool --> Train
        Backend --> Prom["Prometheus (:9090)"]
        Bento --> Prom
        Prom --> Grafana["Grafana Dashboards (:3001)"]
        Backend --> ReplitFE["[External] Replit Frontend"]
    end
```

## Data Ingestion & Manifest Normalization

1. **Schema Standardization**: Both MVTec AD and Casting datasets are parsed into a unified schema:
   `[image_path, source_dataset, product_category, original_label, binary_label, label, split]`
2. **Binary Classification**:
   - `0 = OK`
   - `1 = DEFECT`
3. **Domain Preservation**: Metadata preserves source dataset and product category to enable multi-slice evaluation.
4. **Stratification**: Data is split into isolated Train (70%), Validation (15%), and Test (15%) partitions with zero cross-split leakage.
