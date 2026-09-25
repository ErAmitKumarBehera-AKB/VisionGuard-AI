from typing import Any, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResNet50DefectDetector(nn.Module):

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        hidden_dim: int = 256,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if pretrained:
            try:
                weights = ResNet50_Weights.DEFAULT
                self.backbone = resnet50(weights=weights)
                logger.info("Loaded pretrained ResNet-50 weights (IMAGENET1K).")
            except Exception as e:
                logger.warning(f"Could not download pretrained weights ({e}). Initializing without pretrained weights.")
                self.backbone = resnet50(weights=None)
        else:
            self.backbone = resnet50(weights=None)

        in_features = self.backbone.fc.in_features

        layers: list[nn.Module] = [
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(hidden_dim, num_classes)
        ])

        self.backbone.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def freeze_backbone(self, unfreeze_layers: Optional[list[str]] = None) -> None:
        unfreeze = set(unfreeze_layers or ["fc"])
        for name, param in self.backbone.named_parameters():
            should_unfreeze = any(layer_name in name for layer_name in unfreeze)
            param.requires_grad = should_unfreeze

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Backbone frozen. Trainable params: {trainable_params:,} / {total_params:,}")

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all model parameters for full fine-tuning.")


def build_resnet50_model(config: Optional[dict[str, Any]] = None) -> ResNet50DefectDetector:
    cfg = config or {}
    model_cfg = cfg.get("model", {})
    classifier_cfg = model_cfg.get("classifier", {})

    return ResNet50DefectDetector(
        num_classes=classifier_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        dropout_rate=classifier_cfg.get("dropout_rate", 0.3),
        hidden_dim=classifier_cfg.get("hidden_dim", 256),
        use_batchnorm=classifier_cfg.get("use_batchnorm", True),
    )
