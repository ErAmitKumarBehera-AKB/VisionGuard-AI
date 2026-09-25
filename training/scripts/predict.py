#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.src.inference.predictor import DefectPredictor
from training.src.utils.logging import get_logger, setup_logging

logger = get_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defect inspection inference on an image.")
    parser.add_argument("image_path", type=str, help="Path to input image file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="training/artifacts/checkpoints/best_model.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--threshold", type=float, default=0.50, help="Defect decision threshold.")
    parser.add_argument("--product", type=str, default=None, help="Product category metadata.")
    parser.add_argument("--source", type=str, default=None, help="Source dataset metadata.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    img_path = Path(args.image_path)
    if not img_path.is_file():
        logger.error(f"Image not found: {img_path}")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    cp_arg = str(checkpoint_path) if checkpoint_path.is_file() else None
    if cp_arg is None:
        logger.warning(f"Checkpoint {checkpoint_path} not found. Running with initialized ResNet-50 for demonstration.")

    predictor = DefectPredictor(
        checkpoint_path=cp_arg,
        defect_threshold=args.threshold,
    )

    result = predictor.predict(
        image_input=img_path,
        product_category=args.product,
        source_dataset=args.source,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
