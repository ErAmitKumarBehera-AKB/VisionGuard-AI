#!/usr/bin/env bash
set -e

echo "======================================================================"
echo " Training ResNet-50 Visual Defect Classifier"
echo "======================================================================"

MANIFEST="ml/data/manifests/unified_manifest.parquet"
if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found. Running dataset preparation first..."
    python ml/scripts/prepare_dataset.py --config ml/configs/dataset.yaml
fi

python ml/scripts/train.py --config ml/configs/training.yaml --model-config ml/configs/model.yaml "$@"
