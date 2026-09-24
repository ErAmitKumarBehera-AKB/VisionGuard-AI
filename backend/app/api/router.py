from fastapi import APIRouter
from .v1.inspection import router as inspection_router
from .v1.feedback import router as feedback_router
from .v1.system import router as system_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(inspection_router)
api_v1_router.include_router(feedback_router)
api_v1_router.include_router(system_router)
