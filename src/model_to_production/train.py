"""Train and evaluate the reproducible model artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET, make_dataset


def chronological_split(frame, train_fraction: float = 0.7):
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be >= 0.5 and < 1")
    ordered = frame.sort_values("event_time").reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )


def train(output: Path, metrics_path: Path | None = None) -> dict:
    bundle = make_dataset()
    train_frame, eval_frame = chronological_split(bundle.frame)
    model = build_model()
    model.fit(train_frame[FEATURES], train_frame[TARGET])
    probabilities = model.predict_proba(eval_frame[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "model_version": "purchase-intent-logreg-v1",
        "features": FEATURES,
        "split": "chronological_70_30",
        "train_rows": len(train_frame),
        "evaluation_rows": len(eval_frame),
        "evaluation_start": eval_frame["event_time"].min().isoformat(),
        "evaluation_end": eval_frame["event_time"].max().isoformat(),
        "accuracy": round(float(accuracy_score(eval_frame[TARGET], predictions)), 6),
        "precision": round(float(precision_score(eval_frame[TARGET], predictions)), 6),
        "recall": round(float(recall_score(eval_frame[TARGET], predictions)), 6),
        "roc_auc": round(float(roc_auc_score(eval_frame[TARGET], probabilities)), 6),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, output)
    if metrics_path is None:
        metrics_path = output.with_name("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.metrics), indent=2))


if __name__ == "__main__":
    main()
