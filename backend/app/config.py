from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    BENTOML_SERVICE_URL: str = "http://localhost:3000"
    BENTOML_TIMEOUT_SECONDS: float = 10.0

    MODEL_CHECKPOINT_PATH: str = "training/artifacts/checkpoints/best_model.pt"

    DATABASE_URL: str = "sqlite:///./backend/app/inspection.db"

    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
    ]

    LOW_CONFIDENCE_THRESHOLD: float = 0.80
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    IMAGE_STORAGE_PATH: str = "backend/app/storage/images"
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "visioninspect"
    JWT_SECRET: str = ""
    JWT_EXPIRE_MINUTES: int = 60
    BOOTSTRAP_ADMIN_EMAIL: str = ""
    BOOTSTRAP_ADMIN_PASSWORD: str = ""
    BOOTSTRAP_SUPERVISOR_EMAIL: str = ""
    BOOTSTRAP_SUPERVISOR_PASSWORD: str = ""

    @property
    def image_storage_dir(self) -> Path:
        return Path(self.IMAGE_STORAGE_PATH)


settings = Settings()
