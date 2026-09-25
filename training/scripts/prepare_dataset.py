#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.src.data.casting_loader import CastingLoader
from training.src.data.dataset_builder import UnifiedDatasetBuilder
from training.src.data.mvtec_loader import MVTecLoader
from training.src.data.splitter import DatasetSplitter
from training.src.utils.logging import get_logger, setup_logging

logger = get_logger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and prepare unified manufacturing dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/dataset.yaml",
        help="Path to dataset configuration YAML file.",
    )
    parser.add_argument("--mvtec-dir", type=str, default=None, help="Override MVTec AD root directory.")
    parser.add_argument("--casting-dir", type=str, default=None, help="Override Casting dataset root directory.")
    parser.add_argument("--output-csv", type=str, default=None, help="Output manifest CSV path.")
    parser.add_argument("--output-parquet", type=str, default=None, help="Output manifest Parquet path.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.is_file():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Config file not found at {config_path}. Using environment variables or defaults.")

    raw_cfg = cfg.get("raw_data", {})
    storage_cfg = cfg.get("storage", {})
    split_cfg = cfg.get("splitting", {})

    mvtec_dir = (
        args.mvtec_dir
        or os.getenv("MVTEC_DATA_DIR")
        or raw_cfg.get("mvtec", {}).get("root_dir", "training/data/raw/mvtec")
    )
    casting_dir = (
        args.casting_dir
        or os.getenv("CASTING_DATA_DIR")
        or raw_cfg.get("casting", {}).get("root_dir", "training/data/raw/casting")
    )

    out_csv = (
        args.output_csv
        or storage_cfg.get("manifest_csv", "training/data/manifests/unified_manifest.csv")
    )
    out_parquet = (
        args.output_parquet
        or storage_cfg.get("manifest_parquet", "training/data/manifests/unified_manifest.parquet")
    )

    logger.info("Initializing dataset loaders...")
    mvtec_categories = raw_cfg.get("mvtec", {}).get("categories", ["cable", "screw", "metal_nut", "transistor"])
    mvtec_loader = MVTecLoader(root_dir=mvtec_dir, categories=mvtec_categories)
    casting_loader = CastingLoader(root_dir=casting_dir)

    mvtec_available = mvtec_loader.is_available()
    casting_available = casting_loader.is_available()

    if not mvtec_available and not casting_available:
        logger.error(
            "\n" + "=" * 70 + "\n"
            "DATASET NOTICE: No raw dataset images found!\n\n"
            "Please download the datasets and place them in the following paths:\n"
            f"  1. MVTec AD: {Path(mvtec_dir).resolve()}\n"
            "     Expected categories: cable/, screw/, metal_nut/, transistor/\n"
            f"  2. Casting Product Image Data: {Path(casting_dir).resolve()}\n"
            "     Expected folders: ok_front/, def_front/\n"
            "=" * 70
        )
        sys.exit(1)

    loaders = [mvtec_loader, casting_loader]
    builder = UnifiedDatasetBuilder(loaders=loaders)

    logger.info("Building and validating unified dataset records...")
    df = builder.build_manifest(validate_files=True)

    if len(df) == 0:
        logger.error("No valid image records could be assembled. Aborting.")
        sys.exit(1)

    logger.info("Performing stratified, leakage-safe train/val/test splitting...")
    splitter = DatasetSplitter(
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        random_seed=split_cfg.get("random_seed", 42),
        stratify_cols=split_cfg.get("stratify_columns"),
    )
    df_split = splitter.split(df)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_split.to_csv(out_csv, index=False)
    df_split.to_parquet(out_parquet, index=False)

    logger.info(f"Unified manifest saved to:\n  - CSV: {out_csv}\n  - Parquet: {out_parquet}")
    logger.info(f"Total samples: {len(df_split):,}")
    logger.info(f"Class breakdown:\n{df_split['binary_label'].value_counts().to_string()}")
    logger.info(f"Split breakdown:\n{df_split['split'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
