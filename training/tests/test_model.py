import torch
import pytest

from training.src.models.resnet50 import ResNet50DefectDetector, build_resnet50_model


def test_resnet50_instantiation():
    model = ResNet50DefectDetector(num_classes=2, pretrained=False, dropout_rate=0.2)
    assert isinstance(model, torch.nn.Module)
    assert model.num_classes == 2


def test_resnet50_forward_shape():
    model = ResNet50DefectDetector(num_classes=2, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_probabilities(dummy_input)

    assert logits.shape == (2, 2)
    assert probs.shape == (2, 2)
    prob_sums = probs.sum(dim=1).numpy()
    assert pytest.approx(prob_sums[0], 0.001) == 1.0
    assert pytest.approx(prob_sums[1], 0.001) == 1.0


def test_freeze_backbone():
    model = ResNet50DefectDetector(num_classes=2, pretrained=False)
    model.freeze_backbone(unfreeze_layers=["fc"])

    fc_trainable = all(p.requires_grad for p in model.backbone.fc.parameters())
    assert fc_trainable is True

    conv1_trainable = any(p.requires_grad for p in model.backbone.conv1.parameters())
    assert conv1_trainable is False

    model.unfreeze_all()
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable is True


def test_model_factory():
    cfg = {
        "model": {
            "pretrained": False,
            "classifier": {
                "num_classes": 2,
                "dropout_rate": 0.4,
                "hidden_dim": 128,
                "use_batchnorm": True,
            }
        }
    }
    model = build_resnet50_model(cfg)
    assert isinstance(model, ResNet50DefectDetector)
