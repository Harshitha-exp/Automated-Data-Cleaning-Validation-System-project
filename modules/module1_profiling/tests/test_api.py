"""
API tests for Person 4's FastAPI app — uses FastAPI's TestClient (no
network, no running server needed).
"""
import io

import pandas as pd
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_profile_endpoint_returns_full_report():
    df = pd.DataFrame({
        "id": range(1, 11),
        "email": [f"user{i}@example.com" for i in range(10)],
        "amount": [10.0, 20.0, 15.0, 30.0, 50.0, 12.0, 80.0, 9.0, 11.0, 1000.0],
    })
    files = {"file": ("sample.csv", _csv_bytes(df), "text/csv")}
    resp = client.post("/profile", files=files)
    assert resp.status_code == 200

    body = resp.json()
    assert body["n_rows"] == 10
    assert body["n_columns"] == 3
    assert body["metadata"]["columns"]["id"]["semantic"] == "id"
    assert "missing" in body["statistics"]
    assert "missing_heatmap" in body["visualisations"]
    assert body["errors"] == []


def test_profile_endpoint_rejects_unsupported_extension():
    files = {"file": ("sample.txt", b"not a real dataset", "text/plain")}
    resp = client.post("/profile", files=files)
    assert resp.status_code == 400


def test_profile_endpoint_skip_visualisations_flag():
    df = pd.DataFrame({"id": range(1, 6), "val": [1, 2, 3, 4, 5]})
    files = {"file": ("sample.csv", _csv_bytes(df), "text/csv")}
    resp = client.post("/profile?include_visualisations=false", files=files)
    assert resp.status_code == 200
    assert resp.json()["visualisations"] == {}
