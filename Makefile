
SHELL := /bin/bash
PYTHON := /home/msi/.venvs/tcs_project/bin/python
PIP := /home/msi/.venvs/tcs_project/bin/pip
PYTEST := /home/msi/.venvs/tcs_project/bin/pytest

.PHONY: help setup prepare-data train evaluate test lint serve backend qc docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make setup          - Setup Python virtual environment and install dependencies"
	@echo "  make prepare-data   - Run dataset ingestion and manifest generation"
	@echo "  make train          - Train ResNet-50 binary classifier on unified dataset"
	@echo "  make evaluate       - Run comprehensive & domain-aware evaluation"
	@echo "  make test           - Run full test suite (unit & integration tests)"
	@echo "  make serve          - Start BentoML model serving service"
	@echo "  make backend        - Start FastAPI backend application"
	@echo "  make qc             - Start Streamlit Human-in-the-Loop QC interface"
	@echo "  make docker-up      - Launch multi-container system (backend, serving, qc, prometheus, grafana)"
	@echo "  make docker-down    - Stop and remove Docker containers"
	@echo "  make clean          - Clean temporary caches and test artifacts"

setup:
	@bash scripts/setup.sh

prepare-data:
	$(PYTHON) ml/scripts/prepare_dataset.py --config ml/configs/dataset.yaml

train:
	$(PYTHON) ml/scripts/train.py --config ml/configs/training.yaml

evaluate:
	$(PYTHON) ml/scripts/evaluate.py --checkpoint ml/artifacts/checkpoints/best_model.pt

test:
	$(PYTEST) ml/tests backend/tests -v

lint:
	$(PYTHON) -m ruff check ml/ backend/ serving/ edge/ qc/

serve:
	bentoml serve serving/service:svc --port 3000

backend:
	$(PYTHON) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

qc:
	$(PYTHON) -m streamlit run qc/app.py --server.port 8501

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
