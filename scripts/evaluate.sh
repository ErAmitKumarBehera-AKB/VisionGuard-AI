#!/usr/bin/env bash
set -e

echo "======================================================================"
echo " Running Domain-Aware & Product-Aware Model Evaluation"
echo "======================================================================"

python ml/scripts/evaluate.py --checkpoint ml/artifacts/checkpoints/best_model.pt "$@"
