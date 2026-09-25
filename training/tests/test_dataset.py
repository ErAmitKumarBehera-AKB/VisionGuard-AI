from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from training.src.data.casting_loader import CastingLoader
from training.src.data.dataset_builder import BaseDatasetLoader, ManufacturingDataset, UnifiedDatasetBuilder
from training.src.data.mvtec_loader import MVTecLoader
from training.src.data.splitter import DatasetSplitter
from training.src.data.validation import DatasetValidator


@pytest.fixture
def synthetic_mvtec_dir(tmp_path: Path) -> Path:
    mvtec_root = tmp_path / "mvtec"
    category = "metal_nut"

    train_good = mvtec_root / category / "train" / "good"
    train_good.mkdir(parents=True)
    img = Image.new("RGB", (64, 64), color="gray")
    img.save(train_good / "train_001.png")

    test_good = mvtec_root / category / "test" / "good"
    test_good.mkdir(parents=True)
    img.save(test_good / "test_good_001.png")

    test_defect = mvtec_root / category / "test" / "scratch"
    test_defect.mkdir(parents=True)
    defect_img = Image.new("RGB", (64, 64), color="red")
    defect_img.save(test_defect / "scratch_001.png")

    return mvtec_root


@pytest.fixture
def synthetic_casting_dir(tmp_path: Path) -> Path:
    casting_root = tmp_path / "casting"
    ok_dir = casting_root / "ok_front"
    def_dir = casting_root / "def_front"
    ok_dir.mkdir(parents=True)
    def_dir.mkdir(parents=True)

    img = Image.new("RGB", (64, 64), color="blue")
    img.save(ok_dir / "cast_ok_001.jpeg")

    img_def = Image.new("RGB", (64, 64), color="yellow")
    img_def.save(def_dir / "cast_def_001.jpeg")

    return casting_root


def test_image_validator(tmp_path: Path):
    valid_img_path = tmp_path / "test.png"
    Image.new("RGB", (64, 64), color="white").save(valid_img_path)

    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not a real image header")

    assert DatasetValidator.validate_image_file(valid_img_path) is True
    assert DatasetValidator.validate_image_file(corrupt_path) is False
    assert DatasetValidator.validate_image_file(tmp_path / "nonexistent.png") is False


def test_mvtec_loader(synthetic_mvtec_dir: Path):
    loader = MVTecLoader(root_dir=synthetic_mvtec_dir, categories=["metal_nut"])
    assert loader.is_available() is True

    records = loader.load()
    assert len(records) == 3

    labels = {r["binary_label"] for r in records}
    assert labels == {"OK", "DEFECT"}

    for r in records:
        assert r["source_dataset"] == "mvtec"
        assert r["product_category"] == "metal_nut"
        if r["original_label"] == "good":
            assert r["binary_label"] == "OK"
            assert r["label"] == 0
        else:
            assert r["binary_label"] == "DEFECT"
            assert r["label"] == 1


def test_casting_loader(synthetic_casting_dir: Path):
    loader = CastingLoader(root_dir=synthetic_casting_dir)
    assert loader.is_available() is True

    records = loader.load()
    assert len(records) == 2
    for r in records:
        assert r["source_dataset"] == "casting"
        assert r["product_category"] == "casting_impeller"
        assert r["binary_label"] in ["OK", "DEFECT"]


def test_unified_dataset_builder(synthetic_mvtec_dir: Path, synthetic_casting_dir: Path):
    loaders = [
        MVTecLoader(synthetic_mvtec_dir, categories=["metal_nut"]),
        CastingLoader(synthetic_casting_dir),
    ]
    builder = UnifiedDatasetBuilder(loaders=loaders)
    df = builder.build_manifest(validate_files=True)

    assert len(df) == 5
    assert set(df["binary_label"].unique()) == {"OK", "DEFECT"}
    assert set(df["source_dataset"].unique()) == {"mvtec", "casting"}


def test_dataset_splitter():
    data = []
    for i in range(20):
        is_defect = i % 2 == 1
        data.append({
            "image_path": f"/mock/path/img_{i}.png",
            "source_dataset": "mvtec" if i < 10 else "casting",
            "product_category": "cable" if i < 10 else "casting_impeller",
            "original_label": "bad" if is_defect else "good",
            "binary_label": "DEFECT" if is_defect else "OK",
            "label": 1 if is_defect else 0,
        })
    df = pd.DataFrame(data)

    splitter = DatasetSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
    df_split = splitter.split(df)

    assert "split" in df_split.columns
    assert set(df_split["split"].unique()) == {"train", "val", "test"}
    assert len(df_split[df_split["split"] == "train"]) > 0
    assert len(df_split[df_split["split"] == "val"]) > 0
    assert len(df_split[df_split["split"] == "test"]) > 0
