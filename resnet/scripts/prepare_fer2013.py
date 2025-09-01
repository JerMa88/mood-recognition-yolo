#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import kagglehub
from PIL import Image


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def read_fer_csv(csv_path: str) -> List[Tuple[str, str, List[int]]]:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pixels = list(map(int, r["pixels"].split()))
            label = EMOTIONS[int(r["emotion"]) if "emotion" in r else int(r["label"])]
            usage = r.get("Usage", r.get("usage", "Training"))
            rows.append((label, usage, pixels))
    return rows


def write_images(rows: List[Tuple[str, str, List[int]]], out_dir: str, val_ratio: float = 0.25):
    # Split: use provided Usage for test; for Training split into train/val (75/25)
    train_items: Dict[str, List[List[int]]] = {e: [] for e in EMOTIONS}
    test_items: Dict[str, List[List[int]]] = {e: [] for e in EMOTIONS}
    for label, usage, pixels in rows:
        if usage.lower().startswith("test"):
            test_items[label].append(pixels)
        else:
            train_items[label].append(pixels)

    # Create train/val
    val_items: Dict[str, List[List[int]]] = {e: [] for e in EMOTIONS}
    for e in EMOTIONS:
        items = train_items[e]
        n = len(items)
        n_val = int(n * val_ratio)
        val_items[e] = items[:n_val]
        train_items[e] = items[n_val:]

    # Write images to disk
    for split, mapping in [("train", train_items), ("val", val_items), ("test", test_items)]:
        for label, lst in mapping.items():
            base = Path(out_dir) / split / label
            base.mkdir(parents=True, exist_ok=True)
            for idx, pix in enumerate(lst):
                img = Image.fromarray(
                    (Image.frombytes("L", (48, 48), bytes(pix))).tobytes(),
                    mode="L",
                    size=(48, 48),
                )
                # The above approach is a bit awkward; simpler:
                img = Image.fromarray(
                    (Image.eval(Image.frombytes("L", (48, 48), bytes(pix)), lambda p: p)).convert("L")
                )
                img.save(base / f"{idx}.png")


def main(args: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Download and prepare FER2013 using kagglehub")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parsed = parser.parse_args(args)

    path = kagglehub.dataset_download("msambare/fer2013")
    print("Path to dataset files:", path)

    # The dataset contains fer2013.csv
    csv_path = os.path.join(path, "fer2013.csv")
    rows = read_fer_csv(csv_path)
    write_images(rows, parsed.out_dir, val_ratio=parsed.val_ratio)
    print("Finished preparing FER2013 in ImageFolder format under", parsed.out_dir)


if __name__ == "__main__":
    main()

