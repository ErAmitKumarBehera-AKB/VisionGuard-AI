# Streamlit Human-in-the-Loop Quality Control (QC)

This application provides a dedicated Human-in-the-Loop (HITL) interface for manufacturing quality control operators.

---

## Core Capabilities

1. **Inspection Feed & Triage**: Displays live inspection predictions from the FastAPI backend and BentoML serving microservice.
2. **Low-Confidence Alert**: Filters uncertain inspections (`confidence < 80%`) for immediate human verification.
3. **Interactive Label Correction**: Operators view the high-resolution component image, inspect model predictions, toggle ground truth (`OK` vs `DEFECT`), and annotate defect characteristics.
4. **Retraining Loop (DVC)**: Corrections are captured in the feedback repository, preparing curated samples for versioned retraining.

---

## Running Locally

```bash
streamlit run qc/app.py --server.port 8501
```
