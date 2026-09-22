import os
import argparse
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


mlflow.set_tracking_uri("file:./mlruns")


def build_resnet50_classifier(num_classes=2):
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet50(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


def train_model(epochs=5, batch_size=16, lr=0.001, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mlflow.set_experiment("Visual_Quality_Inspection")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_train_dir = "data/raw/train"
    raw_test_dir = "data/raw/test"
    feedback_dir = "data/feedback"

    if not os.path.exists(raw_train_dir) or not os.path.exists(raw_test_dir):
        raise ValueError("Dataset directory not found. Please run generate_data.py first.")

    train_dataset = datasets.ImageFolder(root=raw_train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=raw_test_dir, transform=val_transform)

    if sorted(train_dataset.classes) != ["defective", "ok"]:
        raise ValueError(f"Dataset must contain ['defective', 'ok']; found {train_dataset.classes}")

    feedback_datasets = []
    if os.path.exists(feedback_dir):
        subdirs = [d for d in os.listdir(feedback_dir) if os.path.isdir(os.path.join(feedback_dir, d))]
        if len(subdirs) > 0:
            try:
                fb_data = datasets.ImageFolder(root=feedback_dir, transform=train_transform)
                if len(fb_data) > 0:
                    feedback_datasets.append(fb_data)
                    print(f"Incorporated {len(fb_data)} feedback images into retraining dataset.")
            except Exception as e:
                print(f"Could not load feedback directory: {e}")

    if feedback_datasets:
        combined_train_dataset = ConcatDataset([train_dataset] + feedback_datasets)
    else:
        combined_train_dataset = train_dataset

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(combined_train_dataset) > batch_size else False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_resnet50_classifier(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    with mlflow.start_run(run_name="resnet50-defect-classifier") as run:
        print(f"Logging experiment run: {run.info.run_id}")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("architecture", "ResNet-50")
        mlflow.log_param("class_names", ",".join(train_dataset.classes))

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)

            epoch_loss = running_loss / len(combined_train_dataset)
            epoch_acc = correct / total if total > 0 else 0

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", acc)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)

        print(f"\nEvaluation Results:")
        print(f"Loss: {test_loss:.4f} Acc: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")

        model_path = os.path.join(model_dir, "model.pth")
        payload = {
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "classes": train_dataset.classes,
            "architecture": "resnet50"
        }
        torch.save(payload, model_path)
        print(f"Saved PyTorch weights to {model_path}")

        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="ResNet50_Casting_Inspection",
                serialization_format="pickle"
            )
            print("Model logged and registered in MLflow registry.")
        except Exception as exc:
            print(f"MLflow model logging skipped: {exc}")

    return test_loss, acc, precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 Defect Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
