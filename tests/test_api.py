"""
Integration tests for FastAPI endpoints.
"""

import pytest
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "hardware" in data


def test_stream_status_endpoint(client):
    response = client.get("/api/v1/stream/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_running" in data
    assert "worker_concurrency" in data


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_metrics" in data
    assert "queue_metrics" in data


def test_image_upload_inspection(client):
    # Create test synthetic image in memory
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    img_bytes = encoded.tobytes()

    response = client.post(
        "/api/v1/inspect/upload",
        files={"file": ("test_part.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["PASS", "FAIL"]
    assert "anomaly_score" in data
    assert "latency_ms" in data
    assert "annotated_image_base64" in data
