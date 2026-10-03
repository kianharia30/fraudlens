.DEFAULT_GOAL := help
.PHONY: help setup lock download-data validate-data data train train-only evaluate export-demo-pool thresholds figures readme test cov lint format typecheck serve app benchmark synthetic-demo docker clean

PYTHON_SYS ?= python3.11
VENV       ?= .venv
BIN        := $(VENV)/bin
PY         := $(BIN)/python
# API port; override with `make serve PORT=8010`
PORT       ?= 8000

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

$(PY):
	$(PYTHON_SYS) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

setup: $(PY)  ## Create venv, install package + dev tools, install pre-commit hooks
	$(PY) -m pip install -e ".[dev]"
	mkdir -p data/raw
	@if [ -d .git ]; then $(BIN)/pre-commit install; else echo "Not a git repo yet: run 'git init' then '$(BIN)/pre-commit install'"; fi

lock:  ## Freeze the exact environment to requirements.lock
	{ echo "# Full transitive lock for Python 3.11. Regenerate with: make lock"; $(BIN)/pip freeze --exclude-editable; } > requirements.lock

download-data:  ## Download IEEE-CIS via the Kaggle CLI (needs a Kaggle API token)
	$(PY) scripts/download_data.py

validate-data:  ## Check data/raw contains the IEEE-CIS training files
	$(PY) scripts/validate_data.py

data: validate-data  ## Build features + time-based train/valid/test parquet splits
	$(PY) scripts/build_dataset.py

train: data  ## Build data, train baseline + LightGBM, calibrate, pick thresholds, update README
	$(PY) scripts/train.py
	$(PY) scripts/update_readme.py

train-only:  ## Retrain from existing processed splits (skips `make data`)
	$(PY) scripts/train.py
	$(PY) scripts/update_readme.py

evaluate:  ## Re-evaluate the latest saved model on the held-out test block
	$(PY) scripts/evaluate.py

export-demo-pool:  ## Re-export the app's demo pool from the latest model
	$(PY) scripts/export_demo_pool.py

thresholds:  ## Re-tune thresholds on validation after editing costs/budgets (no retraining)
	$(PY) scripts/retune_thresholds.py
	$(PY) scripts/update_readme.py

figures:  ## Re-render docs/figures from the latest model (no retraining)
	$(PY) scripts/render_figures.py

readme:  ## Fill README results from docs/metrics.json and docs/benchmark.json
	$(PY) scripts/update_readme.py

test:  ## Run the test suite (no real data needed)
	$(BIN)/pytest

cov:  ## Run tests with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

lint:  ## Ruff + black check + mypy
	$(BIN)/ruff check .
	$(BIN)/black --check .
	$(BIN)/mypy

format:  ## Auto-fix lint and format
	$(BIN)/ruff check --fix .
	$(BIN)/black .

typecheck:  ## mypy only
	$(BIN)/mypy

serve:  ## Run the FastAPI scoring service on :8000 (PORT=... to change)
	$(BIN)/uvicorn fraudlens.service.api:app --host 0.0.0.0 --port $(PORT)

app:  ## Run the Streamlit demo on :8501
	$(BIN)/streamlit run src/fraudlens/app/streamlit_app.py

benchmark:  ## Latency benchmark (1,000 requests) against a running API (`make serve` first)
	$(PY) scripts/benchmark.py --url http://localhost:$(PORT)
	$(PY) scripts/update_readme.py

SYN := configs/config.synthetic.yaml
synthetic-demo:  ## No data? Run the whole pipeline on fake data, then open the app on it
	$(PY) scripts/make_synthetic_data.py
	$(PY) scripts/build_dataset.py --config $(SYN)
	$(PY) scripts/train.py --config $(SYN)
	FRAUDLENS_CONFIG=$(SYN) $(BIN)/streamlit run src/fraudlens/app/streamlit_app.py

docker:  ## Build and run API + app with docker compose (needs a trained model in artifacts/)
	docker compose up --build

clean:  ## Remove caches (keeps data and model artefacts)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
