#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import requests


def read_metadata(csv_path: str) -> List[Tuple[str, str, str]]:
    """Read GLDv2-style CSV with columns: id, landmark_id, url.

    Returns list of (id, landmark_id, url).
    """
    rows: List[Tuple[str, str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "landmark_id", "url"}
        if not required.issubset(reader.fieldnames or {}):
            raise ValueError(f"CSV must contain columns: {required}")
        for r in reader:
            rows.append((r["id"], r["landmark_id"], r["url"]))
    return rows


def select_small(rows: List[Tuple[str, str, str]], max_classes: int, max_per_class: int) -> Dict[str, List[Tuple[str, str]]]:
    by_class: Dict[str, List[Tuple[str, str]]] = {}
    for img_id, landmark_id, url in rows:
        if landmark_id not in by_class:
            if len(by_class) >= max_classes:
                continue
            by_class[landmark_id] = []
        if len(by_class[landmark_id]) < max_per_class:
            by_class[landmark_id].append((img_id, url))
    return by_class


def download_one(session: requests.Session, url: str, out_path: Path, timeout: int = 15) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        r = session.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return False
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def split_train_val(items: List[Tuple[str, str]], val_ratio: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    n = len(items)
    n_val = max(1, int(n * val_ratio)) if n > 0 else 0
    return items[n_val:], items[:n_val]


def main(args: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Prepare a small GLDv2 subset with downloads by URL")
    parser.add_argument("--metadata-csv", required=True, help="CSV with columns id,landmark_id,url")
    parser.add_argument("--out-dir", default="data", help="Output root directory")
    parser.add_argument("--max-classes", type=int, default=100)
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=16)
    parsed = parser.parse_args(args)

    rows = read_metadata(parsed.metadata_csv)
    by_class = select_small(rows, parsed.max_classes, parsed.max_per_class)

    train_tasks: List[Tuple[str, str, Path]] = []
    val_tasks: List[Tuple[str, str, Path]] = []
    for cls, items in by_class.items():
        train_items, val_items = split_train_val(items, parsed.val_ratio)
        for img_id, url in train_items:
            out_path = Path(parsed.out_dir) / "train" / cls / f"{img_id}.jpg"
            train_tasks.append((url, cls, out_path))
        for img_id, url in val_items:
            out_path = Path(parsed.out_dir) / "val" / cls / f"{img_id}.jpg"
            val_tasks.append((url, cls, out_path))

    all_tasks = train_tasks + val_tasks
    print(f"Classes: {len(by_class)} | train imgs: {len(train_tasks)} | val imgs: {len(val_tasks)}")

    ok = 0
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=parsed.workers) as ex:
            futs = [ex.submit(download_one, session, url, out_path) for (url, _cls, out_path) in all_tasks]
            for fut in as_completed(futs):
                if fut.result():
                    ok += 1
    print(f"Downloaded {ok}/{len(all_tasks)} images")

    print("Done. Folder structure ready for training:")
    print(os.path.join(parsed.out_dir, "train/<class>/*.jpg"))
    print(os.path.join(parsed.out_dir, "val/<class>/*.jpg"))


if __name__ == "__main__":
    main()

