#!/usr/bin/env bash
set -e

echo "======================================================================"
echo " Visual Quality Inspection System - Automated Setup"
echo "======================================================================"

PYTHON_BIN=""
for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3.11 or higher is required."
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"

if command -v uv >/dev/null 2>&1; then
    echo "Found uv. Installing dependencies with uv..."
    uv pip install -r requirements.txt
else
    echo "uv not found. Installing with standard pip..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

mkdir -p ml/data/raw/mvtec ml/data/raw/casting ml/data/manifests \
         ml/artifacts/checkpoints ml/artifacts/exported ml/artifacts/reports \
         backend/app/storage/images

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env configuration file."
fi

echo "======================================================================"
echo " Setup complete! Next steps:"
echo " 1. Place MVTec AD dataset into: ml/data/raw/mvtec/"
echo " 2. Place Casting dataset into: ml/data/raw/casting/"
echo " 3. Run: python ml/scripts/prepare_dataset.py"
echo "======================================================================"
