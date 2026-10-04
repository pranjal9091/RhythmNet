.PHONY: help install test download inspect clean

PYTHON = .venv/bin/python
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest

help:
	@echo "Available make commands:"
	@echo "  make install    - Install project dependencies and editable package"
	@echo "  make download   - Download MIT-BIH Arrhythmia Database from PhysioNet"
	@echo "  make inspect    - Run metadata and record inventory inspection"
	@echo "  make test       - Execute automated test suite"
	@echo "  make clean      - Clean temporary and cache files"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

download:
	$(PYTHON) scripts/download_data.py

inspect:
	$(PYTHON) scripts/inspect_dataset.py

test:
	$(PYTEST) -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
