import os
from typing import Any, Optional
import requests


class QCBackendClient:

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or os.getenv("BACKEND_API_URL", "http://localhost:8000")).rstrip("/")

    def get_summary_stats(self) -> dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/api/v1/inspection/stats/summary", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {
            "total_inspections": 0,
            "defect_count": 0,
            "ok_count": 0,
            "defect_rate_percentage": 0.0,
            "average_latency_ms": 0.0,
            "low_confidence_count": 0,
            "reviewed_count": 0,
        }

    def get_history(
        self,
        page: int = 1,
        page_size: int = 50,
        prediction: Optional[str] = None,
        product_category: Optional[str] = None,
        low_confidence_only: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "low_confidence_only": low_confidence_only,
        }
        if prediction and prediction != "ALL":
            params["prediction"] = prediction
        if product_category and product_category != "ALL":
            params["product_category"] = product_category

        try:
            resp = requests.get(f"{self.base_url}/api/v1/inspection/history", params=params, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"total": 0, "page": page, "page_size": page_size, "items": []}

    def submit_feedback(
        self,
        inspection_id: str,
        operator_label: str,
        comments: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {
            "inspection_id": inspection_id,
            "operator_label": operator_label,
            "comments": comments or "",
        }
        resp = requests.post(f"{self.base_url}/api/v1/feedback", json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_feedback_export(self) -> dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/api/v1/feedback/export", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"total_records": 0, "items": [], "ready_for_dvc_ingestion": False}
