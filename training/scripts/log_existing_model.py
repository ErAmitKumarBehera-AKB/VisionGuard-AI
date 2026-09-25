#!/usr/bin/env python3

from pathlib import Path
import sys
import yaml
import torch
import mlflow
import mlflow.pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.src.models.resnet50 import build_resnet50_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = PROJECT_ROOT / "training/artifacts/checkpoints/best_model.pt"
MODEL_CONFIG = PROJECT_ROOT / "training/configs/model.yaml"
TRAINING_CONFIG = PROJECT_ROOT / "training/configs/training.yaml"


def main():
    print("Loading configuration...")

    with open(MODEL_CONFIG, "r") as f:
        model_cfg = yaml.safe_load(f) or {}

    with open(TRAINING_CONFIG, "r") as f:
        training_cfg = yaml.safe_load(f) or {}

    config = {
        **model_cfg,
        **training_cfg,
    }

    print("Loading checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint metrics: {checkpoint['metrics']}")

    print("Building ResNet-50 model...")
    model = build_resnet50_model(config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model weights loaded successfully.")

    experiment_name = config.get("mlflow", {}).get(
        "experiment_name",
        "manufacturing-defect-inspection-384",
    )

    registered_model_name = config.get("mlflow", {}).get(
        "registered_model_name",
        "ManufacturingDefectResNet50",
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="resnet50-384-epoch24-existing-checkpoint"):

        mlflow.log_params({
            "architecture": "resnet50",
            "image_size": 384,
            "batch_size": 16,
            "epochs_trained": 30,
            "best_epoch": checkpoint["epoch"],
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "checkpoint": str(CHECKPOINT),
        })

        metrics = checkpoint["metrics"]

        mlflow.log_metrics({
            "best_val_accuracy": metrics["accuracy"],
            "best_val_defect_precision": metrics["defect_precision"],
            "best_val_defect_recall": metrics["defect_recall"],
            "best_val_defect_f1": metrics["defect_f1"],
            "best_val_roc_auc": metrics["roc_auc"],
            "best_val_ok_specificity": metrics["ok_specificity"],
        })

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=registered_model_name,
            serialization_format="pickle",
        )

        print()
        print("=" * 60)
        print("MLFLOW LOGGING COMPLETE")
        print("=" * 60)
        print(f"Experiment: {experiment_name}")
        print(f"Registered model: {registered_model_name}")
        print(f"Best epoch: {checkpoint['epoch']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
