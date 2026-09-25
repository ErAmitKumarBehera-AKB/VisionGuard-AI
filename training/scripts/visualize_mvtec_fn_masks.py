from pathlib import Path
import math

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/run/media/msi/CROSS/TCS_project")

FN_CSV = (
    ROOT /
    "ml/artifacts/reports/384_analysis/"
    "false_negatives.csv"
)

OUTPUT = (
    ROOT /
    "ml/artifacts/reports/384_analysis/"
    "mvtec_false_negatives_with_masks.png"
)


def find_mask(image_path):

    image_path = Path(image_path)

    parts = image_path.parts

    try:
        mvtec_idx = parts.index("mvtec")
    except ValueError:
        return None

    category = parts[mvtec_idx + 1]
    split = parts[mvtec_idx + 2]
    defect_type = parts[mvtec_idx + 3]
    filename = image_path.stem

    if split != "test":
        return None

    mask_path = (
        ROOT /
        "ml/data/raw/mvtec" /
        category /
        "ground_truth" /
        defect_type /
        f"{filename}_mask.png"
    )

    if mask_path.exists():
        return mask_path

    return None


df = pd.read_csv(FN_CSV)

df = df[
    df["source_dataset"]
    .astype(str)
    .str.lower()
    == "mvtec"
].copy()

print(f"MVTec false negatives: {len(df)}")

if df.empty:
    raise RuntimeError("No MVTec false negatives found.")


items = []

for _, row in df.iterrows():

    image_path = Path(row["image_path"])
    mask_path = find_mask(image_path)

    if mask_path is None:
        print(
            f"Mask not found: "
            f"{image_path}"
        )
        continue

    items.append({
        "image_path": image_path,
        "mask_path": mask_path,
        "category": row["product_category"],
        "probability": float(
            row["defect_probability"]
        ),
    })


print(f"MVTec FNs with masks: {len(items)}")


if not items:
    raise RuntimeError(
        "No matching MVTec ground-truth masks found."
    )


cell_w = 360
cell_h = 390

cols = 3
rows = len(items)

sheet = Image.new(
    "RGB",
    (
        cols * cell_w,
        rows * cell_h
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)


for i, item in enumerate(items):

    try:

        image = Image.open(
            item["image_path"]
        ).convert("RGB")

        mask = Image.open(
            item["mask_path"]
        ).convert("L")

        image = image.resize(
            (320, 300)
        )

        mask = mask.resize(
            (320, 300)
        )

        mask_np = np.array(mask)

        mask_binary = (
            mask_np > 0
        )

        mask_rgb = np.zeros(
            (300, 320, 3),
            dtype=np.uint8
        )

        mask_rgb[
            mask_binary
        ] = [255, 255, 255]

        mask_image = Image.fromarray(
            mask_rgb
        )

        overlay = image.copy()

        overlay_np = np.array(
            overlay
        )

        overlay_np[
            mask_binary
        ] = (
            0.35 *
            overlay_np[
                mask_binary
            ]
            +
            0.65 *
            np.array(
                [255, 0, 0]
            )
        ).astype(np.uint8)

        overlay = Image.fromarray(
            overlay_np
        )

        x = 20
        y = i * cell_h + 10

        sheet.paste(
            image,
            (x, y)
        )

        sheet.paste(
            mask_image,
            (cell_w + x, y)
        )

        sheet.paste(
            overlay,
            (2 * cell_w + x, y)
        )

        text_y = y + 305

        label = (
            f"{item['category']} | "
            f"P(defect)="
            f"{item['probability']:.4f}"
        )

        draw.text(
            (x, text_y),
            label,
            fill="black"
        )

        mask_area = (
            mask_binary.sum()
            /
            mask_binary.size
            *
            100
        )

        draw.text(
            (x, text_y + 22),
            f"Defect mask area: "
            f"{mask_area:.3f}%",
            fill="black"
        )

        draw.text(
            (x, text_y + 44),
            Path(
                item["image_path"]
            ).name,
            fill="black"
        )

    except Exception as e:

        print(
            f"Failed: "
            f"{item['image_path']} -> {e}"
        )


sheet.save(OUTPUT)

print()
print(f"Saved: {OUTPUT}")
