"""Small in-process observability surface for the reference service."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class Monitor:
    def __init__(self, trace_path: Path = Path("traces/requests.jsonl")):
        self.trace_path = trace_path
        self.request_count = 0
        self.error_count = 0
        self.latencies_ms: list[float] = []

    def record(self, *, features: dict, probability: float, model_version: str) -> str:
        request_id = str(uuid.uuid4())
        self.request_count += 1
        event = {
            "request_id": request_id,
            "timestamp": time.time(),
            "model_version": model_version,
            "outcome": "success",
            "probability": round(probability, 6),
            "feature_keys": sorted(features),
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        return request_id

    def observe_latency(self, milliseconds: float) -> None:
        self.latencies_ms.append(milliseconds)

    def summary(self) -> dict:
        ordered = sorted(self.latencies_ms)
        p50 = ordered[len(ordered) // 2] if ordered else 0.0
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "latency_ms_p50": round(p50, 3),
        }
