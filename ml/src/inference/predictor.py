import io
import time
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from ..models.resnet50 import ResNet50DefectDetector
from ..utils.device import get_device
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DefectPredictor:

    DEFAULT_IMAGE_SIZE = 384
    CLASSES = ["OK", "DEFECT"]

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        model: Optional[ResNet50DefectDetector] = None,
        device: str = "auto",
        model_version: str = "v1.0.0",
        defect_threshold: float = 0.50,
        image_size: int = 384,
    ) -> None:
        self.device = get_device(device)
        self.model_version = model_version
        self.defect_threshold = defect_threshold
        self.image_size = image_size

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_model_from_checkpoint(checkpoint_path)
        else:
            logger.warning("No checkpoint or model provided. Initializing baseline ResNet-50 architecture.")
            self.model = ResNet50DefectDetector(num_classes=2, pretrained=False).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model_from_checkpoint(self, path: str | Path) -> ResNet50DefectDetector:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")

        model = ResNet50DefectDetector(num_classes=2, pretrained=False)
        checkpoint = torch.load(p, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        logger.info(f"Predictor loaded model weights from {p}")
        return model

    def _to_pil_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        product_category: Optional[str] = None,
        source_dataset: Optional[str] = None,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()

        pil_img = self._to_pil_image(image_input)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        p_ok = float(probs[0])
        p_defect = float(probs[1])

        if p_defect >= self.defect_threshold:
            prediction = "DEFECT"
            confidence = p_defect
        else:
            prediction = "OK"
            confidence = p_ok

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2),
            "probabilities": {
                "OK": round(p_ok, 4),
                "DEFECT": round(p_defect, 4),
            },
        }

        if product_category:
            result["product_category"] = product_category
        if source_dataset:
            result["source_dataset"] = source_dataset

        return result
