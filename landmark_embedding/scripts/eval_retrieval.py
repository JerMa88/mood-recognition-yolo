#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import time

from landmark_embedding.infer import InferConfig, build_infer_loader, compute_embeddings
from landmark_embedding.models import create_model


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return a @ b.t()


def average_precision(sorted_labels: np.ndarray, query_label: int, k: int | None = None) -> float:
    if k is not None:
        sorted_labels = sorted_labels[:k]
    relevant = (sorted_labels == query_label).astype(np.int32)
    n_relevant = relevant.sum()
    if n_relevant == 0:
        return 0.0
    cum_hits = 0
    precisions = []
    for i, rel in enumerate(relevant, start=1):
        if rel:
            cum_hits += 1
            precisions.append(cum_hits / i)
    return float(np.mean(precisions))


def recall_at_k(sorted_labels: np.ndarray, query_label: int, k: int) -> float:
    topk = sorted_labels[:k]
    return float((topk == query_label).any())


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval mAP@k using cosine similarity")
    parser.add_argument("--gallery-dir", required=True, help="ImageFolder gallery (e.g., data/train)")
    parser.add_argument("--query-dir", required=True, help="ImageFolder queries (e.g., data/val)")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--weights", required=True, help="Checkpoint .pt containing model state")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=5, help="Compute mAP@k and Recall@k")
    parser.add_argument("--use-faiss", action="store_true", help="Force FAISS for kNN (falls back if unavailable)")
    parser.add_argument("--topk", type=int, default=100, help="Neighbors to retrieve (FAISS)")
    parsed = parser.parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    model = create_model(arch=parsed.arch, embedding_dim=parsed.embedding_dim, pretrained=False)
    ckpt = torch.load(parsed.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # Load gallery and query
    gal_loader = build_infer_loader(InferConfig(parsed.gallery_dir, parsed.img_size, parsed.batch_size, parsed.num_workers))
    qry_loader = build_infer_loader(InferConfig(parsed.query_dir, parsed.img_size, parsed.batch_size, parsed.num_workers))

    gal_emb, gal_labels, _ = compute_embeddings(model, gal_loader, device)
    qry_emb, qry_labels, _ = compute_embeddings(model, qry_loader, device)

    # Normalize for cosine similarity
    gal = torch.nn.functional.normalize(gal_emb, dim=1).cpu().numpy().astype("float32")
    qry = torch.nn.functional.normalize(qry_emb, dim=1).cpu().numpy().astype("float32")
    gal_labels_np = gal_labels.numpy()
    qry_labels_np = qry_labels.numpy()

    # Try FAISS if requested or available
    use_faiss = parsed.use_faiss
    faiss = None
    if use_faiss:
        try:
            import faiss  # type: ignore
        except Exception:
            use_faiss = False
    if not parsed.use_faiss:  # auto-detect if not forced
        try:
            import faiss  # type: ignore
            faiss  # noqa
            use_faiss = True
        except Exception:
            use_faiss = False

    start = time.time()
    if use_faiss:
        import faiss  # type: ignore

        d = gal.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product == cosine since normalized
        index.add(gal)
        D, I = index.search(qry, min(parsed.topk, gal.shape[0]))  # (Nq, topk)
        sorted_idx = I
        backend = f"FAISS(IndexFlatIP, topk={sorted_idx.shape[1]})"
    else:
        sims = (qry @ gal.T)  # numpy cosine since normalized
        sorted_idx = np.argsort(-sims, axis=1)
        backend = "numpy (cosine)"
    elapsed = time.time() - start

    aps = []
    recalls = []
    for i in range(sorted_idx.shape[0]):
        ranked_labels = gal_labels_np[sorted_idx[i]]
        ap = average_precision(ranked_labels, qry_labels_np[i], k=parsed.k)
        r = recall_at_k(ranked_labels, qry_labels_np[i], k=parsed.k)
        aps.append(ap)
        recalls.append(r)

    mean_ap = float(np.mean(aps)) if aps else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    print(
        f"Backend: {backend} | Elapsed: {elapsed:.2f}s | "
        f"mAP@{parsed.k}: {mean_ap:.4f} | Recall@{parsed.k}: {mean_recall:.4f}"
    )


if __name__ == "__main__":
    main()
