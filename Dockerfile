FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home finguard
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install .  && python -m finguard.models.train --out /app/artifacts/models
ENV FINGUARD_MODEL_DIR=/app/artifacts/models \
    FINGUARD_POLICY_PATH=/app/configs/policies/committee_v1.yaml
USER finguard
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "finguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
