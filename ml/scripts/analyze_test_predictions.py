#!/usr/bin/env python3

from pathlib import Path
import sys
import yaml
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.models.resnet50 import build_resnet50_model
from ml.src.data.dataset_builder import ManufacturingDataset
from ml.src.training.train import get_transforms
from ml.src.utils.device import get_device
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = PROJECT_ROOT / "ml/artifacts/checkpoints/best_model.pt"
MANIFEST = PROJECT_ROOT / "ml/data/manifests/unified_manifest.parquet"
MODEL_CONFIG = PROJECT_ROOT / "ml/configs/model.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "ml/configs/training.yaml"

OUTPUT_DIR = PROJECT_ROOT / "ml/artifacts/reports/384_analysis"


def binary_metrics(y_true, y_pred, probs):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probs = np.asarray(probs)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = None

    return {
        "samples": len(y_true),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_subset(df, predictions, name):
    subset = predictions[predictions["product_category"] == name]
    return binary_metrics(
        subset["label"],
        subset["prediction"],
        subset["defect_probability"],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("384x384 MODEL TEST ANALYSIS")
    print("=" * 70)

    with open(MODEL_CONFIG) as f:
        model_cfg = yaml.safe_load(f) or {}

    with open(TRAINING_CONFIG) as f:
        training_cfg = yaml.safe_load(f) or {}

    config = {
        **model_cfg,
        **training_cfg,
    }

    image_size = config.get("training", {}).get("image_size", 384)

    print("\nLoading test manifest...")

    df = pd.read_parquet(MANIFEST)

    test_df = df[df["split"] == "test"].copy()

    print(f"Test samples: {len(test_df)}")

    device = get_device(
        config.get("training", {}).get("device", "auto")
    )

    print(f"Device: {device}")

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    model = build_resnet50_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = get_transforms(
        image_size=image_size,
        is_training=False,
    )

    dataset = ManufacturingDataset(
        test_df,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    rows = []

    print("\nRunning inference...")

    with torch.no_grad():
        for batch in loader:

            images = batch["image"].to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1].cpu().numpy()

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            labels = batch["label"].cpu().numpy()

            categories = batch["product_category"]

            domains = batch["source_dataset"]

            paths = batch["image_path"]

            for i in range(len(labels)):

                rows.append({
                    "image_path": paths[i],
                    "source_dataset": domains[i],
                    "product_category": categories[i],
                    "label": int(labels[i]),
                    "prediction": int(predictions[i]),
                    "defect_probability": float(probabilities[i]),
                })

    predictions = pd.DataFrame(rows)

    predictions["result"] = "TN"

    predictions.loc[
        (predictions["label"] == 0) &
        (predictions["prediction"] == 1),
        "result"
    ] = "FP"

    predictions.loc[
        (predictions["label"] == 1) &
        (predictions["prediction"] == 0),
        "result"
    ] = "FN"

    predictions.loc[
        (predictions["label"] == 1) &
        (predictions["prediction"] == 1),
        "result"
    ] = "TP"

    overall = binary_metrics(
        predictions["label"],
        predictions["prediction"],
        predictions["defect_probability"],
    )

    print("\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)

    for key, value in overall.items():
        print(f"{key:15}: {value}")

    domain_rows = []

    for domain, group in predictions.groupby("source_dataset"):

        metrics = binary_metrics(
            group["label"],
            group["prediction"],
            group["defect_probability"],
        )

        metrics["domain"] = domain

        domain_rows.append(metrics)

    domain_df = pd.DataFrame(domain_rows)

    print("\n" + "=" * 70)
    print("DOMAIN RESULTS")
    print("=" * 70)

    print(
        domain_df[
            [
                "domain",
                "samples",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "fp",
                "fn",
            ]
        ].to_string(index=False)
    )

    product_rows = []

    for product, group in predictions.groupby("product_category"):

        metrics = binary_metrics(
            group["label"],
            group["prediction"],
            group["defect_probability"],
        )

        metrics["product_category"] = product

        product_rows.append(metrics)

    product_df = pd.DataFrame(product_rows)

    product_df = product_df[
        [
            "product_category",
            "samples",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "specificity",
            "fp",
            "fn",
            "tp",
            "tn",
        ]
    ]

    print("\n" + "=" * 70)
    print("PRODUCT CATEGORY RESULTS")
    print("=" * 70)

    print(
        product_df.to_string(index=False)
    )

    false_negatives = predictions[
        predictions["result"] == "FN"
    ].sort_values(
        "defect_probability"
    )

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES")
    print("=" * 70)

    print(
        false_negatives[
            [
                "source_dataset",
                "product_category",
                "defect_probability",
                "image_path",
            ]
        ].to_string(index=False)
    )

    false_positives = predictions[
        predictions["result"] == "FP"
    ].sort_values(
        "defect_probability",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("FALSE POSITIVES")
    print("=" * 70)

    print(
        false_positives[
            [
                "source_dataset",
                "product_category",
                "defect_probability",
                "image_path",
            ]
        ].to_string(index=False)
    )

    predictions.to_csv(
        OUTPUT_DIR / "all_test_predictions.csv",
        index=False,
    )

    domain_df.to_csv(
        OUTPUT_DIR / "domain_metrics.csv",
        index=False,
    )

    product_df.to_csv(
        OUTPUT_DIR / "product_metrics.csv",
        index=False,
    )

    false_negatives.to_csv(
        OUTPUT_DIR / "false_negatives.csv",
        index=False,
    )

    false_positives.to_csv(
        OUTPUT_DIR / "false_positives.csv",
        index=False,
    )

    threshold_rows = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.02,
    ):

        preds = (
            predictions["defect_probability"] >= threshold
        ).astype(int)

        metrics = binary_metrics(
            predictions["label"],
            preds,
            predictions["defect_probability"],
        )

        metrics["threshold"] = round(
            float(threshold),
            2,
        )

        threshold_rows.append(metrics)

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df = threshold_df[
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "fp",
            "fn",
        ]
    ]

    threshold_df.to_csv(
        OUTPUT_DIR / "threshold_analysis.csv",
        index=False,
    )

    summary = pd.DataFrame([overall])

    summary.to_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("REPORTS SAVED")
    print("=" * 70)

    print(
        OUTPUT_DIR
    )

    print("\nFiles:")

    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {path.name}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
