from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_expected_payload(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "nova"}


def test_root_reports_service_info(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "NOVA"
