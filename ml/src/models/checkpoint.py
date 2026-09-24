import json
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelCheckpointManager:

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[dict[str, Any]] = None,
        filename: str = "best_model.pt",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / filename
        meta_path = checkpoint_path.with_suffix(".json")

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(state, checkpoint_path)

        meta_info = {
            "epoch": epoch,
            "metrics": metrics or {},
            "checkpoint_path": str(checkpoint_path),
            **(metadata or {}),
        }
        with open(meta_path, "w") as f:
            json.dump(meta_info, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path} (epoch {epoch})")
        return checkpoint_path

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str | Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        map_location = device if device else torch.device("cpu")
        checkpoint = torch.load(path, map_location=map_location)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (Epoch: {checkpoint.get('epoch', 'N/A')})")
        return checkpoint
