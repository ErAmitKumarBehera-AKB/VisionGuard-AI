#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.src.training.train import ModelTrainer
from training.src.utils.logging import get_logger, setup_logging

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-50 Defect Detector on unified dataset.")
    parser.add_argument("--config", type=str, default="training/configs/training.yaml", help="Path to training config YAML.")
    parser.add_argument("--model-config", type=str, default="training/configs/model.yaml", help="Path to model config YAML.")
    parser.add_argument("--manifest", type=str, default="training/data/manifests/unified_manifest.parquet", help="Manifest path.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f) or {}

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f) or {}

    merged_cfg = {**model_cfg, **train_cfg}

    if args.epochs:
        merged_cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        merged_cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        merged_cfg["training"]["learning_rate"] = args.lr

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        csv_fallback = manifest_path.with_suffix(".csv")
        if csv_fallback.is_file():
            manifest_path = csv_fallback
        else:
            logger.error(
                f"Dataset manifest not found at {manifest_path}!\n"
                "Please run dataset preparation first:\n"
                "  python training/scripts/prepare_dataset.py"
            )
            sys.exit(1)

    logger.info(f"Loading dataset manifest from {manifest_path}...")
    if manifest_path.suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    trainer = ModelTrainer(config=merged_cfg, manifest_df=df)
    results = trainer.run()

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Best Val Defect Recall: {results['best_val_recall']:.4f}")
    logger.info(f"Saved Checkpoint: {results['best_checkpoint_path']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
