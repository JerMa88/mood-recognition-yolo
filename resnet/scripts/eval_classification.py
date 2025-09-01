#!/usr/bin/env python
from __future__ import annotations

import argparse
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet.model import make_resnet


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate FER classifier on an ImageFolder")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parsed = parser.parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((parsed.img_size, parsed.img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(parsed.data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=parsed.batch_size, shuffle=False, num_workers=parsed.num_workers, pin_memory=True)

    model = make_resnet(parsed.arch, num_classes=len(ds.classes), pretrained=False, in_chans=1)
    ckpt = torch.load(parsed.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval"):
            x = x.to(device)
            logits = model(x)
            y_true.append(y)
            y_pred.append(torch.argmax(logits, dim=1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    report_text = classification_report(y_true, y_pred, digits=4, target_names=ds.classes)
    print(report_text)
    cm = confusion_matrix(y_true, y_pred)
    # Save confusion matrix plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ds.classes, yticklabels=ds.classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig("confusion_matrix.png")
    plt.close(fig)

    # Per-class precision/recall bar charts
    import numpy as np
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(ds.classes))), zero_division=0)

    # Precision bar
    fig_p = plt.figure(figsize=(8, 4))
    x = np.arange(len(ds.classes))
    plt.bar(x, prec, color="#4e79a7")
    plt.xticks(x, ds.classes, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("Per-class Precision")
    plt.tight_layout()
    fig_p.savefig("precision_per_class.png")
    plt.close(fig_p)

    # Recall bar
    fig_r = plt.figure(figsize=(8, 4))
    plt.bar(x, rec, color="#59a14f")
    plt.xticks(x, ds.classes, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("Per-class Recall")
    plt.tight_layout()
    fig_r.savefig("recall_per_class.png")
    plt.close(fig_r)

    # Write per-class metrics CSV
    import csv
    with open("metrics_per_class.csv", "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for idx, name in enumerate(ds.classes):
            writer.writerow([name, f"{prec[idx]:.6f}", f"{rec[idx]:.6f}", f"{f1[idx]:.6f}", int(support[idx])])


if __name__ == "__main__":
    main()
