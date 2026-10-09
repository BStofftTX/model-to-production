# Evidence and boundaries

## Verified in this repository

- Dataset generation is deterministic for a fixed seed.
- The split is chronological: earlier records train the model and later records
  evaluate it.
- Training writes a model artifact and a metrics manifest.
- The API exposes health, readiness, prediction, and monitoring endpoints.
- Predictions produce a trace record with request ID, latency, model version,
  and outcome.
- Tests cover the data contract, chronological split, artifact round-trip, API
  prediction, health endpoints, and a regression for the original split bug.
- Docker builds the artifact before starting the HTTP service.

## Explicitly not verified

- No real-world accuracy, profitability, clinical validity, or customer lift.
- No production cloud deployment has been executed by this repository.
- No real customer data or external credentials are included.
- Monitoring is an implementation reference, not a completed SRE operation.
