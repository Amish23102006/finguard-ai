# On Windows without make, use VS Code: Terminal > Run Task...
PY ?= .venv/bin/python

.PHONY: setup test lint format typecheck check train data api dashboard docker clean

setup:            ## create venv and install dev dependencies
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	cp -n .env.example .env || true
	$(PY) -m pre_commit install || true

test:             ## run the test suite with coverage
	$(PY) -m pytest --cov=finguard --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests

typecheck:
	$(PY) -m mypy

check: lint typecheck test

train:            ## train baseline models and write metrics.json
	$(PY) -m finguard.models.train

data:             ## write seeded demo cases to data/sample
	$(PY) -m finguard.synthetic.generator --out data/sample

api:
	$(PY) -m uvicorn finguard.api.main:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run src/finguard/dashboard/app.py

docker:
	docker compose up --build

clean:
	rm -rf artifacts .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
