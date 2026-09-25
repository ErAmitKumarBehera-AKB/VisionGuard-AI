import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from training.src.inference.predictor import DefectPredictor


@pytest.fixture
def synthetic_image() -> Image.Image:
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_predictor_synthetic_image(synthetic_image: Image.Image):
    predictor = DefectPredictor(checkpoint_path=None, device="cpu")
    result = predictor.predict(
        image_input=synthetic_image,
        product_category="metal_nut",
        source_dataset="mvtec",
    )

    assert "prediction" in result
    assert result["prediction"] in ["OK", "DEFECT"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "latency_ms" in result
    assert result["latency_ms"] > 0
    assert "model_version" in result
    assert result["product_category"] == "metal_nut"
    assert result["source_dataset"] == "mvtec"
    assert "probabilities" in result
    assert "OK" in result["probabilities"]
    assert "DEFECT" in result["probabilities"]


def test_predictor_bytes_input(synthetic_image: Image.Image):
    buf = io.BytesIO()
    synthetic_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    predictor = DefectPredictor(checkpoint_path=None, device="cpu")
    result = predictor.predict(image_input=img_bytes)

    assert result["prediction"] in ["OK", "DEFECT"]
    assert isinstance(result["confidence"], float)


def test_predictor_threshold_tuning(synthetic_image: Image.Image):
    strict_predictor = DefectPredictor(checkpoint_path=None, defect_threshold=0.0, device="cpu")
    res1 = strict_predictor.predict(synthetic_image)
    assert res1["prediction"] == "DEFECT"

    lenient_predictor = DefectPredictor(checkpoint_path=None, defect_threshold=1.0001, device="cpu")
    res2 = lenient_predictor.predict(synthetic_image)
    assert res2["prediction"] == "OK"
