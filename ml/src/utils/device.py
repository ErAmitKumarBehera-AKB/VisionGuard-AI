import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU device: {gpu_name} (CUDA {torch.version.cuda})")
    else:
        logger.info("Using CPU device.")

    return device
