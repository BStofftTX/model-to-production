"""Deterministic synthetic data pipeline and contract validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = [
    "sessions_7d",
    "email_opens_30d",
    "days_since_last_visit",
    "cart_items",
    "prior_orders",
    "support_tickets_30d",
]
TARGET = "purchase_30d"


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    features: list[str]
    target: str


def make_dataset(n: int = 2400, seed: int = 42) -> DatasetBundle:
    """Create deterministic synthetic events with a time-varying signal."""
    if n < 100:
        raise ValueError("n must be at least 100")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame(
        {
            "event_time": dates,
            "sessions_7d": rng.poisson(4, n),
            "email_opens_30d": rng.poisson(7, n),
            "days_since_last_visit": rng.integers(0, 45, n),
            "cart_items": rng.poisson(1, n),
            "prior_orders": rng.poisson(2, n),
            "support_tickets_30d": rng.poisson(0.4, n),
        }
    )
    score = (
        0.35 * frame["sessions_7d"]
        + 0.12 * frame["email_opens_30d"]
        - 0.07 * frame["days_since_last_visit"]
        + 0.55 * frame["cart_items"]
        + 0.18 * frame["prior_orders"]
        - 0.25 * frame["support_tickets_30d"]
        + rng.normal(0, 0.7, n)
    )
    probability = 1 / (1 + np.exp(-(score - score.median()) / 2.2))
    frame[TARGET] = rng.binomial(1, probability)
    validate_dataset(frame)
    return DatasetBundle(frame=frame, features=FEATURES.copy(), target=TARGET)


def validate_dataset(frame: pd.DataFrame) -> None:
    required = {"event_time", *FEATURES, TARGET}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["event_time"].duplicated().any():
        raise ValueError("event_time must be unique")
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("features and target cannot contain nulls")
