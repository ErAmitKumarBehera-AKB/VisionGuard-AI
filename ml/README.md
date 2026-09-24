# Machine Learning Subsystem - Manufacturing Quality Inspection

This package houses the complete ML pipeline for defect classification using transfer learning with **ResNet-50**, **PyTorch**, **MLflow**, and **DVC**.

---

## Architecture Overview

```
ml/
├── configs/          # YAML configurations for datasets, training, and model architecture
├── data/             # Raw datasets, interim caches, processed manifests
├── src/              # Core modular Python packages
│   ├── data/         # Ingestion, validation, MVTec & Casting loaders, splitting
│   ├── models/       # ResNet-50 defect detector architecture & checkpoint manager
│   ├── training/     # Training loop, evaluation, defect recall prioritization, metrics
│   ├── inference/    # Low-latency DefectPredictor
│   └── utils/        # Seeding, device auto-selection (CUDA/CPU), logging
├── scripts/          # CLI entry points for data prep, training, evaluation, inference
├── tests/            # Automated test suite using synthetic fixtures
└── artifacts/        # Checkpoints, exported weights, evaluation reports
```

---

## Key Design Principles

1. **Defect Recall Prioritization**: In industrial manufacturing, shipping a defective part (False Negative) is unacceptable. The loss function applies a penalty weight (default 1.5x) to defect errors, and checkpoint selection monitors validation defect recall.
2. **Domain-Aware & Product-Aware Evaluation**: Evaluates models overall, per source domain (`mvtec` vs `casting`), and per product category (`cable`, `screw`, `metal_nut`, `transistor`, `casting_impeller`) to prevent the model from merely memorizing product textures.
3. **Data Leakage Prevention**: Stratified splitting partitions data into strictly isolated `train`, `val`, and `test` splits based on composite strata keys `(source_dataset, product_category, binary_label)`.
4. **Reproducibility**: Global deterministic seeds set across Python, NumPy, and PyTorch CUDA/CuDNN backends.
5. **Extensibility**: `BaseDatasetLoader` enables onboarding new industrial datasets without refactoring training code.
