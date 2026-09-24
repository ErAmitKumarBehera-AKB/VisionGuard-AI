#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.models.checkpoint import ModelCheckpointManager
from ml.src.models.resnet50 import ResNet50DefectDetector
from ml.src.training.evaluate import domain_aware_evaluation
from ml.src.training.train import get_transforms
from ml.src.utils.device import get_device
from ml.src.utils.logging import get_logger, setup_logging

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate defect detection model on test partition.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ml/artifacts/checkpoints/best_model.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="ml/data/manifests/unified_manifest.parquet",
        help="Path to dataset manifest.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="DEFECT classification threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/artifacts/reports",
        help="Directory to save evaluation reports and plots.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        csv_path = manifest_path.with_suffix(".csv")
        if csv_path.is_file():
            manifest_path = csv_path
        else:
            logger.error(f"Manifest not found at {manifest_path}.")
            sys.exit(1)

    logger.info(f"Loading test split from {manifest_path}...")
    if manifest_path.suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    df_test = df[df["split"] == "test"].copy()
    if len(df_test) == 0:
        logger.error("No test samples found in manifest!")
        sys.exit(1)

    device = get_device("auto")
    model = ResNet50DefectDetector(num_classes=2, pretrained=False)
    checkpoint_mgr = ModelCheckpointManager(checkpoint_path.parent)
    checkpoint_mgr.load_checkpoint(model, checkpoint_path, device=device)
    model.to(device)

    eval_transform = get_transforms(image_size=224, is_training=False)

    logger.info(f"Evaluating {len(df_test)} test samples with threshold {args.threshold}...")
    report = domain_aware_evaluation(
        model=model,
        df_test=df_test,
        transform=eval_transform,
        device=device,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    overall = report["overall"]
    logger.info("=" * 60)
    logger.info("TEST EVALUATION REPORT SUMMARY")
    logger.info(f"Total Test Samples: {report['total_test_samples']}")
    logger.info(f"Accuracy:           {overall['accuracy']:.4f}")
    logger.info(f"Defect Recall:      {overall['defect_recall']:.4f} (Key Manufacturing Metric)")
    logger.info(f"Defect Precision:   {overall['defect_precision']:.4f}")
    logger.info(f"Defect F1-Score:    {overall['defect_f1']:.4f}")
    if overall["roc_auc"]:
        logger.info(f"ROC-AUC:            {overall['roc_auc']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
