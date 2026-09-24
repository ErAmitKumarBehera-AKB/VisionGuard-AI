# Machine Learning Pipeline Specification

## Model Architecture

- **Backbone**: ResNet-50 initialized with `IMAGENET1K_V2` weights.
- **Classification Head**:
  - `Dropout(p=0.3)`
  - `Linear(2048, 256)`
  - `BatchNorm1d(256)`
  - `ReLU(inplace=True)`
  - `Dropout(p=0.15)`
  - `Linear(256, 2)` -> Binary logits `[OK, DEFECT]`
- **Loss Function**: Weighted Cross Entropy with `defect_weight = 1.5` to prioritize Defect Recall.
- **Optimization**: AdamW (`lr=1e-4`, `weight_decay=1e-4`) with Cosine Annealing learning rate schedule.

## Domain-Aware Evaluation

In production quality inspection, global accuracy is misleading if the model performs well on one product (e.g. casting) but misses defects on another (e.g. cable wire defects).
The evaluation pipeline generates:
1. **Overall Test Metrics**: Accuracy, Defect Recall, Defect Precision, Defect F1, ROC-AUC.
2. **Domain Slice Metrics**: MVTec vs Casting.
3. **Product Slice Metrics**: Cable, Screw, Metal Nut, Transistor, Casting Impeller.
4. **Confusion Matrix Heatmap**: Saved to `ml/artifacts/reports/confusion_matrix.png`.

## Model Registry & Promotion Criteria

- **Development**: Models trained during exploratory runs.
- **Candidate**: Model checkpoint achieving `Defect Recall >= 95%` and `Validation F1 >= 90%` on isolated validation split.
- **Production**: Model benchmarked and signed off by lead ML engineer after edge verification.
