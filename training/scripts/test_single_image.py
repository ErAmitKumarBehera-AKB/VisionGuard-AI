import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.src.models.resnet50 import build_resnet50_model


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training/artifacts/checkpoints/best_model_384.pt"
)

IMAGE_SIZE = 384


def main():

    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print(
            "python training/scripts/test_single_image.py "
            "/path/to/image.jpg"
        )
        print()
        print("Example:")
        print(
            "python training/scripts/test_single_image.py "
            "/home/msi/Downloads/image.jpg"
        )
        sys.exit(1)

    image_path = Path(sys.argv[1]).expanduser()

    if not image_path.exists():
        print(f"ERROR: Image not found:")
        print(image_path)
        sys.exit(1)

    if not CHECKPOINT_PATH.exists():
        print("ERROR: Model checkpoint not found:")
        print(CHECKPOINT_PATH)
        sys.exit(1)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("SINGLE IMAGE DEFECT TEST")
    print("=" * 70)

    print(f"Image     : {image_path}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device    : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU       : "
            f"{torch.cuda.get_device_name(0)}"
        )

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    config = {
        "training": {
            "image_size": IMAGE_SIZE,
        }
    }

    model = build_resnet50_model(config)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    ok_probability = probabilities[0].item()
    defect_probability = probabilities[1].item()

    prediction = (
        "DEFECT"
        if defect_probability >= 0.5
        else "OK"
    )

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print(f"Original size : {original_size}")
    print(f"Input size    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print()
    print(
        f"OK probability     : "
        f"{ok_probability * 100:.2f}%"
    )
    print(
        f"DEFECT probability : "
        f"{defect_probability * 100:.2f}%"
    )
    print()
    print(f"PREDICTION         : {prediction}")
    print("-" * 70)


if __name__ == "__main__":
    main()
