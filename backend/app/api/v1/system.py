from fastapi import APIRouter, Depends
try:
    import torch
except ImportError:  # ML inference runs in the separate serving container.
    torch = None

from ...config import settings
from ...auth.security import require_admin
from ...services.bentoml_client import BentoMLClient
from ...utils.mongo import mongo_available

router = APIRouter(prefix="/system", tags=["System"])
bentoml_client = BentoMLClient()


@router.get("/status", summary="Get comprehensive system status")
async def get_system_status(admin=Depends(require_admin)) -> dict:
    bento_health = await bentoml_client.check_health()
    gpu_available = bool(torch and torch.cuda.is_available())
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
            "device_count": torch.cuda.device_count() if gpu_available and torch else 0,
        },
        "database": {
            "primary": "mongodb",
            "url_schema": "mongodb",
            "connected": mongo_available(),
            "name": settings.MONGO_DB,
            "legacy_sqlite_compatibility": True,
        },
    }
