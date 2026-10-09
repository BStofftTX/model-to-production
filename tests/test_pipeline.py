from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from model_to_production.data import FEATURES, make_dataset
from model_to_production.service import ModelService, PredictionRequest, app
from model_to_production.train import chronological_split, train


def test_dataset_is_deterministic():
    assert make_dataset().frame.equals(make_dataset().frame)


def test_split_is_chronological():
    train_frame, eval_frame = chronological_split(make_dataset().frame)
    assert train_frame["event_time"].max() < eval_frame["event_time"].min()


def test_split_rejects_leaky_fraction():
    import pytest

    with pytest.raises(ValueError):
        chronological_split(make_dataset().frame, train_fraction=1.0)


def test_artifact_round_trip_and_prediction(tmp_path: Path):
    artifact = tmp_path / "model.joblib"
    metrics = train(artifact)
    bundle = joblib.load(artifact)
    assert metrics["split"] == "chronological_70_30"
    assert bundle["features"] == FEATURES
    service = ModelService(artifact)
    result = service.predict(
        PredictionRequest(
            sessions_7d=5,
            email_opens_30d=8,
            days_since_last_visit=3,
            cart_items=2,
            prior_orders=1,
            support_tickets_30d=0,
        )
    )
    assert 0 <= result["purchase_probability"] <= 1
    assert result["request_id"]


def test_http_health_and_prediction(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    train(artifact)
    from model_to_production import service as service_module

    monkeypatch.setattr(service_module, "service", ModelService(artifact))
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    response = client.post(
        "/predict",
        json={
            "sessions_7d": 5,
            "email_opens_30d": 8,
            "days_since_last_visit": 3,
            "cart_items": 2,
            "prior_orders": 1,
            "support_tickets_30d": 0,
        },
    )
    assert response.status_code == 200
    assert "model_version" in response.json()


def test_latency_timer_starts_before_model_inference(monkeypatch, tmp_path: Path):
    from model_to_production import service as service_module

    clock_reads = []
    clock = iter([10.0, 10.25])

    def fake_clock():
        value = next(clock)
        clock_reads.append(value)
        return value

    class FakeModel:
        def predict_proba(self, row):
            assert clock_reads == [10.0], "Timer must start before inference"
            return [[0.2, 0.8]]

    class FakeMonitor:
        def __init__(self):
            self.latency = None

        def record(self, **kwargs):
            return "test-request"

        def observe_latency(self, milliseconds):
            self.latency = milliseconds

    fake_monitor = FakeMonitor()
    monkeypatch.setattr(service_module, "monitor", fake_monitor)
    monkeypatch.setattr(service_module.time, "perf_counter", fake_clock)
    service = ModelService(tmp_path / "unused.joblib")
    service.bundle = {
        "model": FakeModel(),
        "features": FEATURES,
        "metrics": {"model_version": "test"},
    }
    result = service.predict(
        PredictionRequest(
            sessions_7d=5,
            email_opens_30d=8,
            days_since_last_visit=3,
            cart_items=2,
            prior_orders=1,
            support_tickets_30d=0,
        )
    )
    assert result["request_id"] == "test-request"
    assert fake_monitor.latency == 250.0
