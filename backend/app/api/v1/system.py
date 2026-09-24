from fastapi import APIRouter
import torch

from ...config import settings
from ...services.bentoml_client import BentoMLClient

router = APIRouter(prefix="/system", tags=["System"])
bentoml_client = BentoMLClient()


@router.get("/status", summary="Get comprehensive system status")
async def get_system_status() -> dict:
    bento_health = await bentoml_client.check_health()
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "backend": {
            "version": "1.0.0",
            "host": settings.BACKEND_HOST,
            "port": settings.BACKEND_PORT,
        },
        "bentoml_service": {
            "configured_url": settings.BENTOML_SERVICE_URL,
            "health": bento_health,
        },
        "hardware_acceleration": {
            "cuda_available": gpu_available,
            "device_name": gpu_name or "CPU",
            "device_count": torch.cuda.device_count() if gpu_available else 0,
        },
        "database": {
            "url_schema": settings.DATABASE_URL.split(":///")[0],
            "connected": True,
        },
    }
