# Landmark Embedding

Minimal Python package for landmark embedding utilities (feature extraction, embeddings, and helpers). See `AGENTS.md` for contributor guidelines.

## Quick Start
- Create a venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (optional dev: `-r requirements-dev.txt`)
- Run checks: `pre-commit run --all-files`
- Run tests: `pytest`

## Project Layout
- `src/landmark_embedding/`: Library code (`__init__.py`, helpers)
- `tests/`: Pytest suite (mirrors package paths)
- `configs/`, `scripts/`, `notebooks/`: Optional; add as needed
- `data/`, `models/`: Local-only; gitignored

## Common Tasks
- Lint: `ruff check .`
- Format: `black . && isort .`
- Coverage: `coverage run -m pytest && coverage html`

## Train
Prepare an ImageFolder dataset:

```
data/
  train/
    classA/ img1.jpg ...
    classB/ ...
  val/
    classA/ ...
    classB/ ...
```

Run training (ResNet-50, ArcFace):

```
python scripts/train.py --data-dir data --out-dir runs/exp --epochs 20 --batch-size 64 --embedding-dim 512 --arch resnet50
```

### Distributed (DDP)
Use torchrun to launch multi-GPU training on a single node:

```
torchrun --standalone --nproc_per_node=2 scripts/train.py --data-dir data --out-dir runs/ddp_exp --epochs 20 --batch-size 64 --embedding-dim 512 --arch resnet50
```
Notes:
- Each process binds to one GPU; set `CUDA_VISIBLE_DEVICES` to select GPUs.
- AMP is enabled by default; disable with `--no-amp`.

## Evaluate Retrieval
Compute mAP@k using cosine similarity, with gallery=`data/train` and queries=`data/val`:

```
python scripts/eval_retrieval.py --gallery-dir data/train --query-dir data/val --weights runs/exp/best.pt --k 5
```

Optional: Use FAISS for faster kNN (recommended for large galleries). Install `faiss-cpu` and enable:

```
pip install faiss-cpu
python scripts/eval_retrieval.py --gallery-dir data/train --query-dir data/val --weights runs/exp/best.pt --k 5 --use-faiss --topk 200
```

## GLDv2 Small Helper
If you have a GLDv2 metadata CSV with columns `id,landmark_id,url`, you can build a small subset and download images concurrently:

```
python scripts/prepare_gldv2_small.py --metadata-csv /path/to/train.csv --out-dir data --max-classes 200 --max-per-class 100 --val-ratio 0.2 --workers 16
```

Notes:
- The script downloads images from provided URLs; some may fail or be unavailable.
- Alternatively, use the Kaggle API to fetch GLDv2/competition metadata, then point `--metadata-csv` to that file.

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
