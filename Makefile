.PHONY: setup train test lint check clean

setup:
	@echo "Creating the local environment and installing dependencies..."
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

train:
	@echo "Running the reproducible leakage experiment..."
	.venv/bin/python -m pytorch_foundations.cli --epochs 40 --seed 42 --output artifacts/metrics.json

test:
	@echo "Running the test suite..."
	.venv/bin/python -m pytest -q

lint:
	@echo "Checking style with Ruff..."
	.venv/bin/ruff check .

check: lint test
	@echo "Lint and test checks completed."

clean:
	@echo "Removing generated artifacts and caches..."
	rm -rf build dist .pytest_cache .ruff_cache artifacts
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
