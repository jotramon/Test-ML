.ONESHELL:

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:             ## Create the virtual environment
	uv venv

.PHONY: install
install:          ## Install all dependencies (runtime + dev + test)
	uv sync --all-extras

.PHONY: install-dev
install-dev:      ## Install runtime + dev extras only
	uv sync --extra dev

.PHONY: install-test
install-test:     ## Install runtime + test extras only
	uv sync --extra test

# --- Local deployment ---

.PHONY: run-api-local
run-api-local:          ## Start the API server locally (port 8000)
	DEPLOYMENT_MODE=local uv run uvicorn challenge.api:app --host 0.0.0.0 --port 8000

.PHONY: training-local
training-local:   ## Run the training job locally (reads data/data.csv, no GCP credentials needed)
	DEPLOYMENT_MODE=local uv run python -m challenge.training

.PHONY: serving-local
serving-local:    ## Run the serving job locally (reads local artifacts, no GCP credentials needed)
	DEPLOYMENT_MODE=local uv run python -m challenge.serving

# --- Tests ---

.PHONY: model-test
model-test:       ## Run model unit tests with coverage
	mkdir reports || true
	uv run pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/model

.PHONY: pipeline-test
pipelines-test:    ## Run pipeline integration tests with coverage
	mkdir reports || true
	uv run pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/pipelines

.PHONY: api-test
api-test:         ## Run API tests with coverage
	mkdir reports || true
	uv run pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/api

STRESS_URL = http://127.0.0.1:8000

.PHONY: stress-test
stress-test:      ## Locust stress test (100 users, 60s, headless) — requires running API at STRESS_URL
	mkdir reports || true
	uv run locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H $(STRESS_URL)
