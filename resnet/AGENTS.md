# Repository Guidelines

## Project Structure & Module Organization
- `src/resnet/`: Core library (ResNet model, training, data utils).
- `tests/`: Pytest unit/integration tests (`test_*.py`).
- `scripts/`: Small task runners (dataset prep, training, evaluation).
- `notebooks/`: Experiments and EDA (keep data paths relative).
- `data/`: Local datasets; use `data/train/`, `data/val/`, `data/test/` (gitignored).
- `models/`: Trained weights/artifacts (gitignored unless tiny samples).
- `assets/`: Sample images, diagrams, and README figures.
- `configs/`: YAML/JSON configs for reproducible runs.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create/activate virtual env.
- `pip install -U pip`: Ensure recent pip.
- `pip install -e .[dev]` or `pip install -r requirements.txt -r requirements-dev.txt`: Install runtime and dev deps.
- `pytest -q`: Run tests.
- `pytest -q --maxfail=1 --disable-warnings`: Faster feedback on failures.
- `coverage run -m pytest && coverage html`: Generate coverage report in `htmlcov/`.

## Coding Style & Naming Conventions
- Python 3.10+ with type hints; prefer `from __future__ import annotations` when useful.
- Formatting: `black .` (88 cols). Linting: `ruff check .` (or `flake8`). Imports: `isort .`.
- Docstrings: NumPy/Google style; include shapes and units.
- Naming: modules/files `snake_case`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- Package entry point: `src/resnet/__init__.py`; avoid top-level side effects.

## Task-Specific Notes (FER2013)
- Data prep: `python scripts/prepare_fer2013.py --out-dir data --val-ratio 0.25`.
- Training: `python scripts/train.py --data-dir data --img-size 48 --arch resnet18 --weighted-sampler`.
- Eval: `python scripts/eval_classification.py --data-dir data/val --weights runs/fer18/best.pt`.

## Testing Guidelines
- Framework: `pytest`; place tests under `tests/` mirroring package paths.
- Naming: `tests/test_<module>.py::Test<ClassOrFunc>` where reasonable.
- Coverage: target ≥80% lines/branches for core logic.
- Use factories/fixtures for images and random seeds (`numpy.random.default_rng(0)`).

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat: add orb embedding`, `fix: bounds check descriptor size`).
- Scope small and focused; include why, not just what.
- PRs: clear description, linked issue, reproduction steps, before/after metrics or screenshots where relevant.
- Checklists: tests added/updated, docs updated, `pre-commit run --all-files` clean.

## Security & Data Handling
- Do not commit secrets or large data; keep `.env`, `data/`, and `models/` gitignored.
- Configure reproducibility: pin versions in `pyproject.toml`/`requirements*.txt`; log seeds/configs.
