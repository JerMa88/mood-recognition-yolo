PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: help install-dev lint format test coverage precommit clean

help:
	@echo "Common targets:"
	@echo "  install-dev  Install dev deps (requirements-dev.txt)"
	@echo "  lint         Run Ruff (lint)"
	@echo "  format       Run Black and isort"
	@echo "  test         Run pytest"
	@echo "  coverage     Run coverage with pytest"
	@echo "  precommit    Run all pre-commit hooks"
	@echo "  clean        Remove caches and build artifacts"

install-dev:
	$(PIP) install -r requirements-dev.txt
	-$(PIP) install -e . || true

lint:
	ruff check .

format:
	isort .
	black .

test:
	pytest -q

coverage:
	coverage run -m pytest
	coverage report -m

precommit:
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist **/__pycache__
