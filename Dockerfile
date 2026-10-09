FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .
RUN mkdir -p artifacts traces && python -m model_to_production.train --output artifacts/model.joblib

EXPOSE 8000
CMD ["uvicorn", "model_to_production.service:app", "--host", "0.0.0.0", "--port", "8000"]
