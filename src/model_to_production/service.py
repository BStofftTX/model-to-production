"""FastAPI inference service with health checks and request traces."""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .monitoring import Monitor

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
monitor = Monitor(Path(os.getenv("TRACE_PATH", "traces/requests.jsonl")))


class PredictionRequest(BaseModel):
    sessions_7d: int = Field(ge=0, le=1000)
    email_opens_30d: int = Field(ge=0, le=1000)
    days_since_last_visit: int = Field(ge=0, le=3650)
    cart_items: int = Field(ge=0, le=100)
    prior_orders: int = Field(ge=0, le=1000)
    support_tickets_30d: int = Field(ge=0, le=100)


class ModelService:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.bundle = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact missing: {self.model_path}")
        self.bundle = joblib.load(self.model_path)

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    def predict(self, request: PredictionRequest) -> dict:
        if not self.ready:
            self.load()
        values = request.model_dump()
        row = pd.DataFrame(
            [[values[name] for name in self.bundle["features"]]],
            columns=self.bundle["features"],
        )
        start = time.perf_counter()
        probability = float(self.bundle["model"].predict_proba(row)[0, 1])
        request_id = monitor.record(
            features=values,
            probability=probability,
            model_version=self.bundle["metrics"]["model_version"],
        )
        monitor.observe_latency((time.perf_counter() - start) * 1000)
        return {
            "request_id": request_id,
            "model_version": self.bundle["metrics"]["model_version"],
            "purchase_probability": round(probability, 6),
            "decision": "high_intent" if probability >= 0.5 else "low_intent",
        }


service = ModelService(MODEL_PATH)
app = FastAPI(title="Model to Production", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    try:
        if not service.ready:
            service.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "model_version": service.bundle["metrics"]["model_version"],
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return service.predict(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/monitoring")
def monitoring():
    return monitor.summary()
