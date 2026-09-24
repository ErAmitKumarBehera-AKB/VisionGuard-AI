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

        crops.append(
            image.crop(
                (
                    left,
                    top,
                    left + crop_size,
                    top + crop_size,
                )
            )
        )

    return crops


def metrics(y_true, probability, threshold=0.50):

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability
        ),
        "specificity": (
            tn / (tn + fp)
            if tn + fp > 0
            else 0
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_df = manifest[
        manifest["split"]
        .astype(str)
        .str.lower()
        == "test"
    ].copy()

    print(
        f"Test samples: {len(test_df)}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = build_resnet50_model(
        config
    )

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
        transforms.Resize(
            (384, 384)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ],
        ),
    ])

    rows = []

    crop_names = [
        "full",
        "center",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ]

    with torch.no_grad():

        for idx, row in test_df.reset_index(
            drop=True
        ).iterrows():

            image_path = Path(
                row["image_path"]
            )

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

                probabilities = (
                    probabilities
                    .detach()
                    .cpu()
                    .numpy()
                )

                label = str(
                    row["binary_label"]
                ).strip().upper()

                if label == "DEFECT":
                    true_label = 1

                elif label == "OK":
                    true_label = 0

                else:
                    raise ValueError(
                        f"Unknown label: {label}"
                    )

                result = {
                    "image_path": str(
                        image_path
                    ),
                    "true_label": true_label,
                    "binary_label": label,
                    "source_dataset": row[
                        "source_dataset"
                    ],
                    "product_category": row[
                        "product_category"
                    ],
                }

                for i, name in enumerate(
                    crop_names
                ):

                    if i < len(probabilities):
                        result[
                            f"prob_{name}"
                        ] = float(
                            probabilities[i]
                        )

                    else:
                        result[
                            f"prob_{name}"
                        ] = np.nan

                rows.append(result)

            except Exception as e:

                print(
                    f"Failed: "
                    f"{image_path} -> {e}"
                )

            if (idx + 1) % 250 == 0:

                print(
                    f"Processed "
                    f"{idx + 1}/"
                    f"{len(test_df)}"
                )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No predictions were generated."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_path = (
        OUTPUT_DIR /
        "fusion_raw_predictions.csv"
    )

    df.to_csv(
        raw_path,
        index=False
    )

    print()
    print(
        f"Saved: {raw_path}"
    )

    y_true = df[
        "true_label"
    ].values

    full = df[
        "prob_full"
    ].values

    local_columns = [
        "prob_center",
        "prob_top_left",
        "prob_top_right",
        "prob_bottom_left",
        "prob_bottom_right",
    ]

    local = df[
        local_columns
    ].values

    local_mean = np.nanmean(
        local,
        axis=1
    )

    max_local = np.nanmax(
        local,
        axis=1
    )

    local_mean = np.where(
        np.isnan(local_mean),
        full,
        local_mean
    )

    max_local = np.where(
        np.isnan(max_local),
        full,
        max_local
    )

    local_filled = np.where(
        np.isnan(local),
        full[:, None],
        local
    )

    all_probs = np.column_stack(
        [
            full,
            local_filled
        ]
    )

    top2_mean = (
        np.sort(
            all_probs,
            axis=1
        )[:, -2:].mean(axis=1)
    )

    strategies = {

        "full_only":
            full,

        "mean_all":
            all_probs.mean(
                axis=1
            ),

        "mean_local":
            local_mean,

        "weighted_70_30":
            (
                0.70 * full
                +
                0.30 * local_mean
            ),

        "weighted_75_25":
            (
                0.75 * full
                +
                0.25 * local_mean
            ),

        "weighted_80_20":
            (
                0.80 * full
                +
                0.20 * local_mean
            ),

        "global_plus_max_60_40":
            (
                0.60 * full
                +
                0.40 * max_local
            ),

        "global_plus_max_70_30":
            (
                0.70 * full
                +
                0.30 * max_local
            ),

        "global_plus_max_80_20":
            (
                0.80 * full
                +
                0.20 * max_local
            ),

        "top2_mean":
            top2_mean,

        "max_all":
            all_probs.max(
                axis=1
            ),
    }

    result_rows = []

    for name, probability in strategies.items():

        m = metrics(
            y_true,
            probability,
            threshold=0.50
        )

        m["strategy"] = name

        result_rows.append(m)

    result = pd.DataFrame(
        result_rows
    )

    result = result[
        [
            "strategy",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "specificity",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]

    result_path = (
        OUTPUT_DIR /
        "fusion_comparison.csv"
    )

    result.to_csv(
        result_path,
        index=False
    )

    print()
    print("=" * 110)
    print(
        "FUSION STRATEGIES @ THRESHOLD 0.50"
    )
    print("=" * 110)

    print(
        result.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {result_path}"
    )


    original_prediction = (
        full >= 0.50
    ).astype(int)

    original_fn = (
        (y_true == 1)
        &
        (original_prediction == 0)
    )

    print()
    print("=" * 110)
    print(
        f"ORIGINAL FULL-IMAGE FALSE NEGATIVES: "
        f"{original_fn.sum()}"
    )
    print("=" * 110)

    for name, probability in strategies.items():

        if name == "full_only":
            continue

        prediction = (
            probability >= 0.50
        ).astype(int)

        recovered = (
            original_fn
            &
            (prediction == 1)
        )

        print(
            f"{name:30s} "
            f"recovers {recovered.sum():2d} / "
            f"{original_fn.sum()} FNs"
        )


    candidate_names = [
        "full_only",
        "weighted_70_30",
        "weighted_75_25",
        "weighted_80_20",
        "global_plus_max_70_30",
        "global_plus_max_80_20",
        "top2_mean",
    ]

    product_rows = []

    for product, group in df.groupby(
        "product_category"
    ):

        indices = group.index

        y = group[
            "true_label"
        ].values

        for name in candidate_names:

            probability = strategies[
                name
            ][indices]

            m = metrics(
                y,
                probability,
                threshold=0.50
            )

            product_rows.append({
                "product_category":
                    product,

                "strategy":
                    name,

                "samples":
                    len(group),

                "accuracy":
                    m["accuracy"],

                "precision":
                    m["precision"],

                "recall":
                    m["recall"],

                "f1":
                    m["f1"],

                "fp":
                    m["fp"],

                "fn":
                    m["fn"],
            })

    product_result = pd.DataFrame(
        product_rows
    )

    product_path = (
        OUTPUT_DIR /
        "fusion_product_comparison.csv"
    )

    product_result.to_csv(
        product_path,
        index=False
    )

    print()
    print(
        f"Saved: {product_path}"
    )

    print()
    print("=" * 110)
    print(
        "FUSION EVALUATION COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
