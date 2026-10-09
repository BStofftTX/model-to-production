# Model to Production

An end-to-end, cloud-ready machine-learning reference implementation.

This project ships a synthetic purchase-intent model through the complete path:

1. deterministic data generation and validation;
2. leakage-resistant time-based training/evaluation;
3. versioned model artifact and metrics;
4. containerized FastAPI inference service;
5. request traces, health checks, and lightweight monitoring;
6. CI tests and a documented failure-and-fix record.

The data is synthetic. This repository makes no customer, clinical, financial,
or production-performance claims.

## Quick start

```bash
uv sync --dev
uv run python -m model_to_production.train
uv run uvicorn model_to_production.service:app --reload
```

Then open `http://127.0.0.1:8000/docs` or run:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"sessions_7d":5,"email_opens_30d":8,"days_since_last_visit":3,"cart_items":2,"prior_orders":1,"support_tickets_30d":0}'
```

## Container path

```bash
docker build -t model-to-production .
docker run --rm -p 8000:8000 model-to-production
```

The image trains the reproducible artifact during the image build, then starts
the API. A deployment workflow template for a managed container service is in
`.github/workflows/deploy-cloud-run.yml`; it is intentionally manual and needs
cloud credentials supplied through GitHub environment secrets.

## Demonstration and verification

After starting the service, open `http://127.0.0.1:8000/docs` to use the interactive API documentation. Submit the sample prediction request above, then inspect:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/monitoring
```

The prediction response identifies the model version and request ID. The monitoring endpoint summarizes request counts and observed latency; the local `traces/requests.jsonl` file records request metadata without raw feature values. These are reference observability features, not a durable production monitoring stack.

For repeatable local checks, run:

```bash
uv sync --locked --dev
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The project deliberately uses synthetic data and does not claim deployment to a live cloud service. The cloud deployment workflow is a template, not proof of a deployed endpoint.

## Evidence

- `docs/EVIDENCE.md` — what is tested and what is not claimed.
- `docs/FAILURE-AND-FIX.md` — the initial failure, diagnosis, repair, and the
  regression test that prevents recurrence.
- `artifacts/` — generated locally/CI and excluded from Git history.
- `traces/` — generated request traces and excluded from Git history.

## Security and limitations

- No real user data, credentials, or external service calls are required.
- The example is a reference implementation, not an autonomous decision-maker.
- Before real deployment: add authentication, encrypted artifact storage,
  secret management, rate limiting, durable metrics, alerting, model registry,
  data contracts, and an approved data-retention policy.
