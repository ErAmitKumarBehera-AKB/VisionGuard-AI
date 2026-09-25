import os
import random
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.src.models.resnet50 import build_resnet50_model

SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
MIN_DELTA = 0.001

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = PROJECT_ROOT / "training/data/manifests/unified_manifest.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "training/artifacts/checkpoints"
REPORT_DIR = PROJECT_ROOT / "training/artifacts/reports/localcrop_384"

BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model_localcrop_384.pt"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_manifest():
    df = pd.read_csv(MANIFEST)

    required = {
        "image_path",
        "binary_label",
        "split",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    df["label"] = (
        df["binary_label"]
        .astype(str)
        .str.upper()
        .map({"OK": 0, "DEFECT": 1})
    )

    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), "binary_label"].unique()
        raise ValueError(f"Unknown labels found: {bad}")

    return df


class LocalCropTrainDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

        self.full_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.55, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        dataset_name = str(
            row.get("dataset", row.get("source", ""))
        ).lower()

        path_lower = str(image_path).lower()

        is_mvtec = (
            "mvtec" in dataset_name
            or "/mvtec/" in path_lower
            or "\\mvtec\\" in path_lower
        )

        if is_mvtec and random.random() < 0.5:
            image = self.local_transform(image)
        else:
            image = self.full_transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class EvalDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            row["image_path"],
        )


def calculate_metrics(labels, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            labels,
            probabilities,
        )
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_probs = []

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_labels.extend(labels.numpy())
        all_probs.extend(
            probs.detach().cpu().numpy()
        )

    labels = np.asarray(all_labels)
    probabilities = np.asarray(all_probs)

    return calculate_metrics(
        labels,
        probabilities,
        threshold=0.5,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    seed_everything(SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("LOCAL-CROP TRAINING - 384x384")
    print("=" * 70)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    df = load_manifest()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print()
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    print()
    print("Train labels:")
    print(train_df["binary_label"].value_counts())

    train_dataset = LocalCropTrainDataset(
        train_df
    )

    val_dataset = EvalDataset(val_df)
    test_dataset = EvalDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    config = {
        "training": {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
        }
    }

    model = build_resnet50_model(config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    start_epoch = 1

    if args.resume:
        print()
        print(f"Loading checkpoint: {args.resume}")

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        start_epoch = (
            checkpoint.get("epoch", 0) + 1
        )

    best_recall = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        model.train()

        running_loss = 0.0
        sample_count = 0

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = model(images)
                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

        scheduler.step()

        train_loss = (
            running_loss / sample_count
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {train_loss:.5f} | "
            f"Val Acc {val_metrics['accuracy']:.5f} | "
            f"Val Precision {val_metrics['precision']:.5f} | "
            f"Val Recall {val_metrics['recall']:.5f} | "
            f"Val F1 {val_metrics['f1']:.5f} | "
            f"Val AUC {val_metrics['roc_auc']:.5f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
            },
        })

        current_recall = val_metrics["recall"]

        if current_recall > (
            best_recall + MIN_DELTA
        ):
            best_recall = current_recall
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "scheduler_state_dict": (
                        scheduler.state_dict()
                    ),
                    "best_val_recall": (
                        best_recall
                    ),
                    "config": config,
                },
                BEST_CHECKPOINT,
            )

            print(
                f"  -> New best checkpoint "
                f"saved: {BEST_CHECKPOINT}"
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print()
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    history_df = pd.DataFrame(history)

    history_path = (
        REPORT_DIR / "training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    print()
    print("FINAL TEST RESULTS")
    print("-" * 70)

    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    results = {
        "experiment": "localcrop_384",
        "best_epoch": best_epoch,
        "best_val_recall": best_recall,
        "test_metrics": test_metrics,
    }

    results_path = (
        REPORT_DIR / "final_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Checkpoint: {BEST_CHECKPOINT}")
    print(f"History:    {history_path}")
    print(f"Results:    {results_path}")


if __name__ == "__main__":
    main()
