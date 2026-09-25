from pathlib import Path
import sys
import yaml
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.models.resnet50 import build_resnet50_model


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "ml/configs/training.yaml"
MANIFEST_PATH = ROOT / "ml/data/manifests/unified_manifest.csv"
CHECKPOINT_PATH = ROOT / "ml/artifacts/checkpoints/best_model_384.pt"

OUTPUT_DIR = ROOT / "ml/artifacts/reports/multicrop_384"


def make_crops(image, source_dataset):

    image = image.convert("RGB")

    if str(source_dataset).lower() == "casting":
        return [image]

    width, height = image.size

    crops = [image]

    crop_size = int(min(width, height) * 0.70)

    if crop_size <= 0:
        return crops

    positions = [
        ("center", 0.50, 0.50),
        ("top_left", 0.30, 0.30),
        ("top_right", 0.70, 0.30),
        ("bottom_left", 0.30, 0.70),
        ("bottom_right", 0.70, 0.70),
    ]

    for _, cx_ratio, cy_ratio in positions:

        cx = int(width * cx_ratio)
        cy = int(height * cy_ratio)

        left = max(
            0,
            min(
                width - crop_size,
                cx - crop_size // 2
            )
        )

        top = max(
            0,
            min(
                height - crop_size,
                cy - crop_size // 2
            )
        )

        crop = image.crop(
            (
                left,
                top,
                left + crop_size,
                top + crop_size,
            )
        )

        crops.append(crop)

    return crops


def main():

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    manifest = pd.read_csv(MANIFEST_PATH)

    test_df = manifest[
        manifest["split"].astype(str).str.lower() == "test"
    ].copy()

    print(f"Test samples: {len(test_df)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = build_resnet50_model(config)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    rows = []

    with torch.no_grad():

        for idx, row in test_df.reset_index(drop=True).iterrows():

            image_path = Path(row["image_path"])

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                source_dataset = str(
                    row["source_dataset"]
                ).strip().lower()

                crops = make_crops(
                    image,
                    source_dataset
                )

                tensors = torch.stack([
                    transform(crop)
                    for crop in crops
                ]).to(device)

                logits = model(tensors)

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )[:, 1]

                probabilities_cpu = (
                    probabilities
                    .detach()
                    .cpu()
                    .numpy()
                )

                full_probability = float(
                    probabilities_cpu[0]
                )

                multicrop_probability = float(
                    probabilities_cpu.max()
                )

                max_crop_index = int(
                    probabilities_cpu.argmax()
                )

                label_text = str(
                    row["binary_label"]
                ).strip().upper()

                if label_text == "DEFECT":
                    true_label = 1
                elif label_text == "OK":
                    true_label = 0
                else:
                    raise ValueError(
                        f"Unknown label: {label_text}"
                    )

                rows.append({
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "binary_label": label_text,
                    "source_dataset": row["source_dataset"],
                    "product_category": row["product_category"],
                    "full_probability": full_probability,
                    "multicrop_probability": multicrop_probability,
                    "max_crop_index": max_crop_index,
                    "num_crops": len(crops),
                })

            except Exception as e:

                print(
                    f"Failed: {image_path} -> {e}"
                )

            if (idx + 1) % 250 == 0:
                print(
                    f"Processed "
                    f"{idx + 1}/{len(test_df)}"
                )

    predictions = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_path = (
        OUTPUT_DIR /
        "multicrop_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False
    )

    print()
    print(
        f"Saved: {predictions_path}"
    )

    print(
        f"Predictions: {len(predictions)}"
    )


    y_true = predictions["true_label"].values

    full_prob = predictions[
        "full_probability"
    ].values

    multi_prob = predictions[
        "multicrop_probability"
    ].values

    print()
    print("=" * 90)
    print("FULL IMAGE vs MULTI-CROP @ THRESHOLD 0.50")
    print("=" * 90)

    for name, prob in [
        ("FULL_IMAGE", full_prob),
        ("MULTI_CROP", multi_prob),
    ]:

        y_pred = (
            prob >= 0.50
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        print()
        print(name)
        print("-" * 50)

        print(
            f"Accuracy:    "
            f"{accuracy_score(y_true, y_pred):.6f}"
        )

        print(
            f"Precision:   "
            f"{precision_score(y_true, y_pred, zero_division=0):.6f}"
        )

        print(
            f"Recall:      "
            f"{recall_score(y_true, y_pred, zero_division=0):.6f}"
        )

        print(
            f"F1:          "
            f"{f1_score(y_true, y_pred, zero_division=0):.6f}"
        )

        print(
            f"ROC-AUC:     "
            f"{roc_auc_score(y_true, prob):.6f}"
        )

        print(
            f"Specificity: "
            f"{tn / (tn + fp):.6f}"
        )

        print(
            f"TN={tn}  FP={fp}  "
            f"FN={fn}  TP={tp}"
        )


    domain_rows = []

    for domain, group in predictions.groupby(
        "source_dataset"
    ):

        y = group["true_label"].values

        for method, column in [
            (
                "full_image",
                "full_probability"
            ),
            (
                "multicrop",
                "multicrop_probability"
            ),
        ]:

            p = group[column].values

            pred = (
                p >= 0.50
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y,
                pred,
                labels=[0, 1]
            ).ravel()

            domain_rows.append({
                "source_dataset": domain,
                "method": method,
                "samples": len(group),
                "accuracy": accuracy_score(
                    y, pred
                ),
                "precision": precision_score(
                    y,
                    pred,
                    zero_division=0
                ),
                "recall": recall_score(
                    y,
                    pred,
                    zero_division=0
                ),
                "f1": f1_score(
                    y,
                    pred,
                    zero_division=0
                ),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            })

    domain_result = pd.DataFrame(
        domain_rows
    )

    domain_result.to_csv(
        OUTPUT_DIR /
        "domain_comparison.csv",
        index=False
    )

    print()
    print("=" * 90)
    print("DOMAIN COMPARISON")
    print("=" * 90)

    print(
        domain_result.to_string(
            index=False
        )
    )


    product_rows = []

    for product, group in predictions.groupby(
        "product_category"
    ):

        y = group["true_label"].values

        for method, column in [
            (
                "full_image",
                "full_probability"
            ),
            (
                "multicrop",
                "multicrop_probability"
            ),
        ]:

            p = group[column].values

            pred = (
                p >= 0.50
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y,
                pred,
                labels=[0, 1]
            ).ravel()

            product_rows.append({
                "product_category": product,
                "method": method,
                "samples": len(group),
                "accuracy": accuracy_score(
                    y, pred
                ),
                "precision": precision_score(
                    y,
                    pred,
                    zero_division=0
                ),
                "recall": recall_score(
                    y,
                    pred,
                    zero_division=0
                ),
                "f1": f1_score(
                    y,
                    pred,
                    zero_division=0
                ),
                "fp": fp,
                "fn": fn,
            })

    product_result = pd.DataFrame(
        product_rows
    )

    product_result.to_csv(
        OUTPUT_DIR /
        "product_comparison.csv",
        index=False
    )

    print()
    print("=" * 90)
    print("PRODUCT COMPARISON")
    print("=" * 90)

    print(
        product_result.to_string(
            index=False
        )
    )


    full_pred = (
        predictions["full_probability"]
        >= 0.50
    )

    multi_pred = (
        predictions["multicrop_probability"]
        >= 0.50
    )

    changed = predictions[
        full_pred != multi_pred
    ].copy()

    changed["full_prediction"] = (
        full_pred[full_pred.index]
        .astype(int)
    )

    changed["multicrop_prediction"] = (
        multi_pred[multi_pred.index]
        .astype(int)
    )

    changed_path = (
        OUTPUT_DIR /
        "changed_predictions.csv"
    )

    changed.to_csv(
        changed_path,
        index=False
    )

    print()
    print(
        f"Decision changes: "
        f"{len(changed)}"
    )

    print(
        f"Saved: {changed_path}"
    )


if __name__ == "__main__":
    main()
