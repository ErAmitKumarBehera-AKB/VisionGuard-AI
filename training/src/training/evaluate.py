import json
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logging import get_logger
from .metrics import calculate_binary_metrics, plot_confusion_matrix

logger = get_logger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["label"].cpu().numpy().tolist()

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            preds = [1 if p >= threshold else 0 for p in probs]

            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.extend(probs)

    metrics = calculate_binary_metrics(all_targets, all_preds, all_probs)
    return metrics


def domain_aware_evaluation(
    model: nn.Module,
    df_test: pd.DataFrame,
    transform: Any,
    device: torch.device,
    batch_size: int = 32,
    threshold: float = 0.5,
    output_dir: str | Path = "ml/artifacts/reports",
) -> dict[str, Any]:
    from ..data.dataset_builder import ManufacturingDataset

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = ManufacturingDataset(df_test, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_domains: list[str] = []
    all_products: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["label"].cpu().numpy().tolist()
            domains = batch["source_dataset"]
            products = batch["product_category"]

            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            preds = [1 if p >= threshold else 0 for p in probs]

            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_domains.extend(domains)
            all_products.extend(products)

    overall_metrics = calculate_binary_metrics(all_targets, all_preds, all_probs)

    cm_counts = [
        [
            overall_metrics["confusion_matrix"]["true_negatives_ok"],
            overall_metrics["confusion_matrix"]["false_positives_defect_alarm"],
        ],
        [
            overall_metrics["confusion_matrix"]["false_negatives_missed_defect"],
            overall_metrics["confusion_matrix"]["true_positives_detected_defect"],
        ],
    ]
    cm_plot_path = out_path / "confusion_matrix.png"
    plot_confusion_matrix(cm_counts, cm_plot_path, title="Overall Defect Detection Confusion Matrix")

    domain_metrics: dict[str, Any] = {}
    unique_domains = sorted(list(set(all_domains)))
    for dom in unique_domains:
        idxs = [i for i, d in enumerate(all_domains) if d == dom]
        if idxs:
            dom_targets = [all_targets[i] for i in idxs]
            dom_preds = [all_preds[i] for i in idxs]
            dom_probs = [all_probs[i] for i in idxs]
            domain_metrics[dom] = calculate_binary_metrics(dom_targets, dom_preds, dom_probs)

    product_metrics: dict[str, Any] = {}
    unique_products = sorted(list(set(all_products)))
    for prod in unique_products:
        idxs = [i for i, p in enumerate(all_products) if p == prod]
        if idxs:
            prod_targets = [all_targets[i] for i in idxs]
            prod_preds = [all_preds[i] for i in idxs]
            prod_probs = [all_probs[i] for i in idxs]
            product_metrics[prod] = calculate_binary_metrics(prod_targets, prod_preds, prod_probs)

    report = {
        "evaluation_threshold": threshold,
        "total_test_samples": len(all_targets),
        "overall": overall_metrics,
        "per_domain": domain_metrics,
        "per_product_category": product_metrics,
    }

    report_json_path = out_path / "evaluation_metrics.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report generated successfully at {report_json_path}")
    logger.info(
        f"Overall Results -> Accuracy: {overall_metrics['accuracy']:.4f} | "
        f"Defect Recall: {overall_metrics['defect_recall']:.4f} | "
        f"Defect Precision: {overall_metrics['defect_precision']:.4f} | "
        f"Defect F1: {overall_metrics['defect_f1']:.4f}"
    )

    return report
