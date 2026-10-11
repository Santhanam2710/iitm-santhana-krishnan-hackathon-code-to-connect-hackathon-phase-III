"""API contract tests with FastAPI's TestClient (idle mode: no background replay)."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["TRIDENT_MODE"] = "idle"


@pytest.fixture(scope="module")
def client():
    from trident.api.app import app

    with TestClient(app) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/healthz").json()["ok"]
    st = client.get("/api/status").json()
    assert st["mode"] == "idle" and "latency_ms" in st


def test_analyze_returns_valid_signals(client):
    r = client.post("/api/analyze", json={"text": "JPMorgan downgraded two notches as commercial real estate losses mount"}).json()
    s = r["signals"][0]
    assert s["entity"]["ticker"] == "JPM" and s["event_classification"]["label"] == "Credit Event" and 1 <= s["impact_score"] <= 10


def test_analyze_publish_flows_to_store_and_modules(client):
    client.post("/api/analyze", json={"text": "Silicon Valley Bank seized by regulators; FDIC takes control", "publish": True})
    rows = client.get("/api/signals?compact=true").json()
    assert any(r["ticker"] == "SIVB" for r in rows)
    assert client.get("/api/rebalancer").json()["last"] is not None


def test_stress_endpoints(client):
    r = client.post("/api/stress/run", json={"event_class": "Credit Event", "impact": 9}).json()
    assert r["pnl"] < 0 and r["cet1_after"] < r["cet1_before"]
    p = client.post("/api/stress/run", json={"preset": "covid_2020"}).json()
    assert "COVID" in p["label"]
    c = client.post("/api/stress/run", json={"event_class": "Credit Event", "impact": 9, "origin_sector": "Banks"}).json()
    assert c["pnl"] <= r["pnl"] + 1e-9  # contagion can only add losses
    assert set(client.get("/api/stress/presets").json()["presets"]) >= {"covid_2020", "svb_2023"}


def test_validation_errors(client):
    assert client.post("/api/analyze", json={"text": ""}).status_code == 422
    assert client.post("/api/stress/run", json={"impact": 42}).status_code == 422


def test_metrics_and_dashboard(client):
    assert "trident_signals_total" in client.get("/metrics").text
    assert "TRIDENT" in client.get("/").text
