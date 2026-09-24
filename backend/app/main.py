from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .api.router import api_v1_router
from .api.v1.secure import router as secure_router
from .config import settings
from .utils.database import init_db
from .utils.mongo import init_mongo, mongo_available
from .utils.logger import get_backend_logger

logger = get_backend_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Visual Quality Inspection API server...")
    logger.info("Initializing database schema...")
    init_db()
    logger.info("Database schema initialized.")
    if mongo_available():
        init_mongo()
        logger.info("MongoDB indexes initialized.")
    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title="Visual Quality Inspection System API",
    description=(
        "Production-style backend for TCS Industry-Aligned Capstone (Use Case B: "
        "Visual Quality Inspection System for Manufacturing). Coordinates ResNet-50 "
        "inference serving via BentoML, inspection logging, human-in-the-loop QC, and metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router)
app.include_router(secure_router, prefix="/api/v1")


@app.get("/health", tags=["Health"], summary="Liveness and readiness check")
def health_check() -> dict:
    return {
        "status": "healthy",
        "service": "manufacturing-quality-inspection-backend",
        "environment": settings.ENVIRONMENT,
    }


@app.get("/metrics", tags=["Observability"], summary="Prometheus metrics scrape target")
def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=(settings.ENVIRONMENT == "development"),
    )
