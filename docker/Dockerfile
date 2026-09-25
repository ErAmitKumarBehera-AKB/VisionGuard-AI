# multi-arch friendly image for x86 and ARM edge devices
FROM python:3.10-slim

ARG TARGETPLATFORM
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=3000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/
COPY mlruns/ /app/mlruns/

EXPOSE 3000 8501 5000

CMD ["bash", "-lc", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////app/mlruns/mlflow.db --default-artifact-root /app/mlruns & bentoml serve src.serve:CastingInspectionService --host 0.0.0.0 --port 3000"]
