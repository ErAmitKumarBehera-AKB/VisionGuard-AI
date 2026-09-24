import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.utils.database import init_db


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    init_db()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "manufacturing-quality-inspection-backend" in data["service"]


def test_metrics_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inspection_total" in response.text or "python_info" in response.text


def test_system_status(client: TestClient):
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "hardware_acceleration" in data


def test_inspection_prediction_flow(client: TestClient, sample_image_bytes: bytes):
    files = {"image": ("test_part.png", sample_image_bytes, "image/png")}
    data = {"product_category": "metal_nut", "source_dataset": "mvtec"}

    response = client.post("/api/v1/inspection/predict", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()

    assert "inspection_id" in payload
    assert payload["prediction"] in ["OK", "DEFECT"]
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["latency_ms"] > 0
    assert payload["product_category"] == "metal_nut"
    assert payload["source_dataset"] == "mvtec"

    inspection_id = payload["inspection_id"]

    get_res = client.get(f"/api/v1/inspection/{inspection_id}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["inspection_id"] == inspection_id
    assert detail["feedback_status"] == "PENDING"

    fb_data = {
        "inspection_id": inspection_id,
        "operator_label": "DEFECT",
        "comments": "Confirmed scratch on surface by QC inspector.",
    }
    fb_res = client.post("/api/v1/feedback", json=fb_data)
    assert fb_res.status_code == 201
    fb_payload = fb_res.json()
    assert fb_payload["inspection_id"] == inspection_id
    assert fb_payload["operator_label"] == "DEFECT"

    get_res2 = client.get(f"/api/v1/inspection/{inspection_id}")
    assert get_res2.status_code == 200
    assert get_res2.json()["operator_label"] == "DEFECT"
    assert get_res2.json()["feedback_status"] in ["CONFIRMED", "CORRECTED"]


def test_inspection_history(client: TestClient):
    response = client.get("/api/v1/inspection/history?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)


def test_inspection_stats_summary(client: TestClient):
    response = client.get("/api/v1/inspection/stats/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_inspections" in data
    assert "defect_rate_percentage" in data


def test_feedback_export(client: TestClient):
    response = client.get("/api/v1/feedback/export")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert "items" in data
