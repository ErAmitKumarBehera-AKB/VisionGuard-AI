import time
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import mlflow
import mlflow.pytorch

from ..data.dataset_builder import ManufacturingDataset
from ..models.checkpoint import ModelCheckpointManager
from ..models.resnet50 import ResNet50DefectDetector, build_resnet50_model
from ..utils.device import get_device
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .evaluate import evaluate_model
from .metrics import calculate_binary_metrics

logger = get_logger(__name__)


def get_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class ModelTrainer:

    def __init__(
        self,
        config: dict[str, Any],
        manifest_df: pd.DataFrame,
        checkpoint_dir: str | Path = "training/artifacts/checkpoints",
    ) -> None:
        self.config = config
        self.manifest_df = manifest_df
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_manager = ModelCheckpointManager(self.checkpoint_dir)

        train_cfg = config.get("train", config.get("training", {}))
        self.seed = train_cfg.get("seed", 42)
        set_seed(self.seed)

        self.device = get_device(train_cfg.get("device", "auto"))
        self.image_size = train_cfg.get("image_size", 224)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.num_workers = train_cfg.get("num_workers", 2)
        self.epochs = train_cfg.get("epochs", 10)
        self.lr = float(train_cfg.get("learning_rate", 1e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.use_amp = train_cfg.get("use_amp", True) and self.device.type == "cuda"
        self.defect_loss_weight = float(train_cfg.get("defect_loss_weight", 1.5))

        self.model = build_resnet50_model(config)
        self.model.to(self.device)

        class_weights = torch.tensor([1.0, self.defect_loss_weight], dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def prepare_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_df = self.manifest_df[self.manifest_df["split"] == "train"]
        val_df = self.manifest_df[self.manifest_df["split"] == "val"]
        test_df = self.manifest_df[self.manifest_df["split"] == "test"]

        train_transform = get_transforms(self.image_size, is_training=True)
        eval_transform = get_transforms(self.image_size, is_training=False)

        train_ds = ManufacturingDataset(train_df, transform=train_transform)
        val_ds = ManufacturingDataset(val_df, transform=eval_transform)
        test_ds = ManufacturingDataset(test_df, transform=eval_transform)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=(self.device.type == "cuda")
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers
        )

        logger.info(
            f"DataLoaders initialized: Train ({len(train_ds)}), Val ({len(val_ds)}), Test ({len(test_ds)})"
        )
        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, dict[str, Any]]:
        self.model.train()
        total_loss = 0.0
        all_targets: list[int] = []
        all_preds: list[int] = []
        all_probs: list[float] = []

        for batch in train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy().tolist()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
            targets = labels.detach().cpu().numpy().tolist()

            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.extend(probs)

        avg_loss = total_loss / len(train_loader.dataset)
        metrics = calculate_binary_metrics(all_targets, all_preds, all_probs)
        return avg_loss, metrics

    def run(self) -> dict[str, Any]:
        train_loader, val_loader, test_loader = self.prepare_dataloaders()

        mlflow_enabled = self.config.get("mlflow", {}).get("enabled", True)
        exp_name = self.config.get("mlflow", {}).get("experiment_name", "manufacturing-defect-inspection")

        if mlflow_enabled:
            mlflow.set_experiment(exp_name)
            run_context = mlflow.start_run()
        else:
            run_context = None

        best_val_recall = -1.0
        best_checkpoint_path = None
        history: list[dict[str, Any]] = []

        try:
            if mlflow_enabled:
                mlflow.log_params({
                    "architecture": "resnet50",
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "optimizer": "adamw",
                    "device": self.device.type,
                    "defect_loss_weight": self.defect_loss_weight,
                    "image_size": self.image_size,
                    "dataset_size_train": len(train_loader.dataset),
                    "dataset_size_val": len(val_loader.dataset),
                })

            logger.info(f"Starting training on {self.device} for {self.epochs} epochs...")

            for epoch in range(1, self.epochs + 1):
                start_time = time.time()
                train_loss, train_metrics = self.train_epoch(train_loader)
                val_metrics = evaluate_model(self.model, val_loader, self.device)
                self.scheduler.step()

                elapsed = time.time() - start_time
                val_recall = val_metrics["defect_recall"]
                val_f1 = val_metrics["defect_f1"]

                logger.info(
                    f"Epoch [{epoch:02d}/{self.epochs:02d}] ({elapsed:.1f}s) - "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Recall: {val_recall:.4f} | "
                    f"Val F1: {val_f1:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                if mlflow_enabled:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "train_defect_recall": train_metrics["defect_recall"],
                        "train_defect_f1": train_metrics["defect_f1"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_defect_recall": val_recall,
                        "val_defect_f1": val_f1,
                        "val_defect_precision": val_metrics["defect_precision"],
                    }, step=epoch)

                epoch_summary = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_metrics": val_metrics,
                }
                history.append(epoch_summary)

                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        filename="best_model.pt",
                        metadata={"config": self.config},
                    )
                    logger.info(f"New best model saved with Defect Recall: {best_val_recall:.4f}")

            logger.info("Evaluating best model on isolated test partition...")
            if best_checkpoint_path and best_checkpoint_path.is_file():
                self.checkpoint_manager.load_checkpoint(self.model, best_checkpoint_path, device=self.device)

            test_metrics = evaluate_model(self.model, test_loader, self.device)
            logger.info(
                f"Isolated Test Results -> Accuracy: {test_metrics['accuracy']:.4f} | "
                f"Defect Recall: {test_metrics['defect_recall']:.4f} | "
                f"Defect Precision: {test_metrics['defect_precision']:.4f} | "
                f"Defect F1: {test_metrics['defect_f1']:.4f}"
            )

            if mlflow_enabled:
                mlflow.log_metrics({
                    "test_accuracy": test_metrics["accuracy"],
                    "test_defect_recall": test_metrics["defect_recall"],
                    "test_defect_precision": test_metrics["defect_precision"],
                    "test_defect_f1": test_metrics["defect_f1"],
                })
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    name="model",
                    registered_model_name=self.config.get("mlflow", {}).get("registered_model_name"),
                    serialization_format="pickle",
                )

            return {
                "best_val_recall": best_val_recall,
                "best_checkpoint_path": str(best_checkpoint_path),
                "test_metrics": test_metrics,
                "history": history,
            }

        finally:
            if run_context:
                mlflow.end_run()
