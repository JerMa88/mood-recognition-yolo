# Facial Expression Recognition (FER2013)

Minimal Python project for facial expression recognition using ResNet on FER2013 (grayscale 48×48). See `AGENTS.md` for contributor guidelines.

## Quick Start
- Create a venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (optional dev: `-r requirements-dev.txt`)
- Run checks: `pre-commit run --all-files`
- Run tests: `pytest`

## Project Layout
- `src/resnet/`: Library code (`model.py`, `train.py`)
- `tests/`: Pytest suite (mirrors package paths)
- `configs/`, `scripts/`, `notebooks/`: Optional; add as needed
- `data/`, `models/`: Local-only; gitignored

## Common Tasks
- Lint: `ruff check .`
- Format: `black . && isort .`
- Coverage: `coverage run -m pytest && coverage html`

## Data: FER2013 via KaggleHub
Download and prepare FER2013 (creates ImageFolder splits with 75/25 train/val and official test):

```
python scripts/prepare_fer2013.py --out-dir data --val-ratio 0.25
```

## Train (ResNet-18, grayscale 1‑ch)
```
python scripts/train.py --data-dir data --out-dir runs/fer18 --epochs 30 --batch-size 256 --img-size 48 --arch resnet18 --weighted-sampler --class-weighted-ce --label-smoothing 0.1
```

### Distributed (DDP)
```
torchrun --standalone --nproc_per_node=2 scripts/train.py --data-dir data --out-dir runs/ddp_fer --epochs 30 --batch-size 256 --img-size 48 --arch resnet18 --weighted-sampler
```

### Distributed (DDP)
Use torchrun to launch multi-GPU training on a single node:

```
torchrun --standalone --nproc_per_node=2 scripts/train.py --data-dir data --out-dir runs/ddp_exp --epochs 20 --batch-size 64 --embedding-dim 512 --arch resnet50
```
Notes:
- Each process binds to one GPU; set `CUDA_VISIBLE_DEVICES` to select GPUs.
- AMP is enabled by default; disable with `--no-amp`.

## Evaluate Classification (val/test)
```
python scripts/eval_classification.py --data-dir data/val --weights runs/fer18/best.pt --arch resnet18 --img-size 48
python scripts/eval_classification.py --data-dir data/test --weights runs/fer18/best.pt --arch resnet18 --img-size 48
```

Artifacts
- Training saves per-epoch confusion matrices to the run directory and updates `confusion_matrix_best.png` on improvements.
- Evaluation writes `confusion_matrix.png` in the working directory.
- Training also logs `history.json` and plots `loss_curve.png` and `accuracy_curve.png` under the run directory.
- Evaluation writes per-class bar charts: `precision_per_class.png` and `recall_per_class.png`.
- CSV logs: training history at `runs/<exp>/history.csv`, and evaluation per-class metrics at `metrics_per_class.csv`.

## Notebook Quickstart
Run an end-to-end training/evaluation workflow directly from a notebook:

- Open `notebooks/FER2013_Quickstart.ipynb` and run all cells.

## Notes on Class Imbalance
- The training script supports `--weighted-sampler` to reduce class bias using inverse-frequency sampling.
- Cross-entropy with label smoothing improves calibration; consider tuning smoothing for best results.

### Kaggle CLI (metadata download)
- Install and authenticate Kaggle CLI:
  - `pip install kaggle`
  - Place your API token at `~/.kaggle/kaggle.json` (chmod 600).
- Download a dataset or competition files (replace placeholders):
  - Dataset: `kaggle datasets download -d <owner>/<dataset-slug> -p data && unzip data/*.zip -d data`
  - Competition: `kaggle competitions download -c <competition-slug> -p data && unzip data/*.zip -d data`
- Then locate the metadata CSV with columns `id,landmark_id,url` and run the prepare script.

## Contributing
- Follow Conventional Commits (e.g., `feat: add orb embedding`)
- Include tests for new behavior; target ≥80% coverage
- Keep PRs focused, with clear descriptions and repro steps

## License
Internal project. Do not distribute without permission.
