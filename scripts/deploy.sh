#!/usr/bin/env bash
set -e

echo "======================================================================"
echo " Launching Production Containerized Inspection System"
echo "======================================================================"

docker compose up --build -d

echo "Services launched:"
echo "  - BentoML Serving: http://localhost:3000"
echo "  - FastAPI Backend: http://localhost:8000 (Docs: /docs)"
echo "  - Streamlit HITL QC: http://localhost:8501"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3001 (admin / admin)"
