from pathlib import Path
import sys
import yaml
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.models.resnet50 import build_resnet50_model


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "ml/configs/training.yaml"
MANIFEST_PATH = ROOT / "ml/data/manifests/unified_manifest.csv"
CHECKPOINT_PATH = ROOT / "ml/artifacts/checkpoints/best_model.pt"
OUTPUT_DIR = ROOT / "ml/artifacts/reports/384_analysis"


def main():

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    manifest = pd.read_csv(MANIFEST_PATH)

    val_df = manifest[
        manifest["split"].astype(str).str.lower() == "val"
    ].copy()

    print(f"Validation samples: {len(val_df)}")

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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_size = config["training"]["image_size"]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    rows = []

    with torch.no_grad():

        for idx, row in val_df.reset_index(drop=True).iterrows():

            image_path = Path(row["image_path"])

            try:

                image = Image.open(image_path).convert("RGB")

                image = transform(image).unsqueeze(0).to(device)

                logits = model(image)

                probability = torch.softmax(
                    logits,
                    dim=1
                )[0, 1].item()

                label_text = str(row["binary_label"]).strip().upper()

                if label_text == "DEFECT":
                    true_label = 1
                elif label_text == "OK":
                    true_label = 0
                else:
                    raise ValueError(
                        f"Unknown binary_label: {row['binary_label']}"
                    )

                rows.append({
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "binary_label": label_text,
                    "defect_probability": probability,
                    "source_dataset": row["source_dataset"],
                    "product_category": row["product_category"],
                })

            except Exception as e:

                print(
                    f"Failed: {image_path}: {e}"
                )

            if (idx + 1) % 500 == 0:
                print(
                    f"Processed {idx + 1}/{len(val_df)}"
                )

    output = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        "all_val_predictions.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print()
    print(f"Saved: {output_path}")
    print(f"Predictions: {len(output)}")

    print()
    print("Validation label distribution:")

    print(
        output["binary_label"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()
