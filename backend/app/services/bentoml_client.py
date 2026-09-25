import asyncio
from pathlib import Path
from typing import Any, Optional
import httpx
from PIL import Image

from ..config import settings
from ..utils.logger import get_backend_logger

logger = get_backend_logger("bentoml_client")


class BentoMLClient:

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or settings.BENTOML_SERVICE_URL).rstrip("/")
        self.timeout = timeout or settings.BENTOML_TIMEOUT_SECONDS
        self._fallback_predictor = None
        self._http_client: Optional[httpx.AsyncClient] = None

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    def _get_fallback_predictor(self):
        if self._fallback_predictor is None:
            try:
                from training.src.inference.predictor import DefectPredictor

                cp = settings.MODEL_CHECKPOINT_PATH
                valid_cp = cp if Path(cp).is_file() else None
                logger.info("Initializing in-process DefectPredictor fallback...")
                self._fallback_predictor = DefectPredictor(checkpoint_path=valid_cp)
            except Exception as e:
                logger.error(f"Failed to initialize fallback predictor: {e}")
                raise
        return self._fallback_predictor

    async def check_health(self) -> dict[str, Any]:
        url = f"{self.base_url}/livez"
        try:
            resp = await self._client().get(url)
            if resp.status_code == 200:
                try:
                    details = resp.json()
                except ValueError:
                    details = {"endpoint": "livez"}
                return {"status": "connected", "details": details}
            return {"status": "degraded", "code": resp.status_code}
        except Exception as e:
            return {"status": "disconnected", "error": str(e), "mode": "in_process_fallback_ready"}

    async def predict(
        self,
        image_bytes: bytes,
        filename: str = "sample.png",
        product_category: Optional[str] = None,
        source_dataset: Optional[str] = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/predict"
        files = {"image": (filename, image_bytes, "image/png")}
        data: dict[str, Any] = {}
        if product_category:
            data["product_category"] = product_category
        if source_dataset:
            data["source_dataset"] = source_dataset

        try:
            resp = await self._client().post(url, files=files, data=data)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"BentoML responded with status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(
                f"BentoML service at {url} unreachable ({e}). "
                "Engaging in-process ResNet-50 fallback predictor."
            )

        predictor = self._get_fallback_predictor()
        return await asyncio.to_thread(
            predictor.predict,
            image_input=image_bytes,
            product_category=product_category,
            source_dataset=source_dataset,
        )
