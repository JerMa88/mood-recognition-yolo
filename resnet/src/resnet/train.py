from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from .data import DataConfig, build_dataloaders
from .model import make_resnet


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "runs/fer"
    arch: str = "resnet18"
    img_size: int = 48
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pretrained: bool = True
    weighted_sampler: bool = True
    num_workers: int = 4
    amp: bool = True
    seed: int = 0
    distributed: bool = False
    dist_backend: str = "nccl"
    label_smoothing: float = 0.1
    class_weighted_ce: bool = False


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return tuple(res)


def setup_distributed(cfg: TrainConfig):
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = cfg.distributed or world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device("cuda", local_rank) if (torch.cuda.is_available()) else torch.device("cpu")
    backend = cfg.dist_backend if torch.cuda.is_available() else "gloo"
    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
    return distributed, rank, local_rank, world_size, device, backend


def cleanup_distributed():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, amp=True, rank=0, criterion=None):
    model.train()
    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False) if rank == 0 else loader
    running_loss = 0.0
    running_top1 = 0.0
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        top1 = accuracy(logits.detach(), y, topk=(1,))[0]
        running_loss += loss.item() * x.size(0)
        running_top1 += top1.item() * x.size(0) / 100.0
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{top1.item():.2f}"})
    n = len(loader.dataset)
    return running_loss / n, 100.0 * running_top1 / n


@torch.no_grad()
def validate(model, loader, device, amp=True, distributed=False) -> Tuple[float, float, str, object]:
    import torch.distributed as dist
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    model.eval()
    ce = nn.CrossEntropyLoss()
    running_loss = 0.0
    preds = []
    targets = []
    is_master = (not distributed) or dist.get_rank() == 0
    iterator = tqdm(loader, desc="Val", leave=False) if is_master else loader
    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=amp):
            logits = model(x)
            loss = ce(logits, y)
        running_loss += loss.item() * x.size(0)
        preds.append(torch.argmax(logits, dim=1).cpu())
        targets.append(y.cpu())

    running_loss = torch.tensor(running_loss, device=device)
    n_local = torch.tensor(sum(t.size(0) for t in targets), device=device, dtype=torch.float64)

    if distributed:
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_local, op=dist.ReduceOp.SUM)

    mean_loss = (running_loss / n_local).item()
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    acc = (y_true == y_pred).mean() * 100.0
    report = classification_report(y_true, y_pred, digits=4)

    fig = None
    if is_master:
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
    return mean_loss, acc, report, fig


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train ResNet for FER2013 emotion classification")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="runs/fer")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--class-weighted-ce", action="store_true")
    parsed = parser.parse_args(args)

    cfg = TrainConfig(
        data_dir=parsed.data_dir,
        out_dir=parsed.out_dir,
        arch=parsed.arch,
        img_size=parsed.img_size,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        pretrained=not parsed.no_pretrained,
        weighted_sampler=parsed.weighted_sampler,
        num_workers=parsed.num_workers,
        amp=not parsed.no_amp,
        seed=parsed.seed,
        distributed=parsed.distributed,
        dist_backend=parsed.dist_backend,
        label_smoothing=parsed.label_smoothing,
        class_weighted_ce=parsed.class_weighted_ce,
    )

    set_seed(cfg.seed)
    distributed, rank, local_rank, world_size, device, backend = setup_distributed(cfg)

    train_loader, val_loader, num_classes, train_sampler, val_sampler = build_dataloaders(
        DataConfig(cfg.data_dir, cfg.img_size, cfg.batch_size, cfg.num_workers, grayscale=True, weighted_sampler=cfg.weighted_sampler),
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    assert num_classes == 7, f"Expected 7 emotion classes, found {num_classes}"

    model = make_resnet(cfg.arch, num_classes=num_classes, pretrained=cfg.pretrained, in_chans=1).to(device)

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, output_device=local_rank if device.type == "cuda" else None)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)

    # Class-weighted cross-entropy option
    class_weights = None
    if cfg.class_weighted_ce:
        from .data import compute_class_weights

        base_ds = train_loader.dataset
        ds = base_ds.dataset if hasattr(base_ds, "dataset") else base_ds
        class_weights = compute_class_weights(ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    best_acc = 0.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_top1 = train_one_epoch(model, train_loader, opt, scaler, device, epoch, cfg.amp, rank=rank, criterion=criterion)
        val_loss, val_acc, report, fig = validate(model, val_loader, device, cfg.amp, distributed=distributed)
        sched.step()

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        if rank == 0:
            state = {
                "epoch": epoch,
                "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if cfg.amp else None,
                "best_acc": best_acc,
                "cfg": cfg.__dict__,
            }
            torch.save(state, os.path.join(cfg.out_dir, "last.pt"))
            if is_best:
                torch.save(state, os.path.join(cfg.out_dir, "best.pt"))
            print(f"Epoch {epoch}/{cfg.epochs} | train {train_loss:.4f}/{train_top1:.2f} | val {val_loss:.4f}/{val_acc:.2f} | best {best_acc:.2f}")
            # Optional: print classification report summary
            print(report)
            # Save confusion matrix plot
            if fig is not None:
                import matplotlib.pyplot as plt
                fig.savefig(os.path.join(cfg.out_dir, f"confusion_matrix_epoch{epoch}.png"))
                if is_best:
                    fig.savefig(os.path.join(cfg.out_dir, "confusion_matrix_best.png"))
                plt.close(fig)

            # Update and save history
            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_top1))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))

            import json
            with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
                json.dump(history, f)

            # Write/append CSV history
            try:
                import csv
                csv_path = os.path.join(cfg.out_dir, "history.csv")
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as cf:
                    writer = csv.writer(cf)
                    if write_header:
                        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
                    writer.writerow([epoch, f"{train_loss:.6f}", f"{train_top1:.4f}", f"{val_loss:.6f}", f"{val_acc:.4f}"])
            except Exception:
                pass

            # Plot loss/accuracy curves
            try:
                import matplotlib.pyplot as plt

                # Loss curve
                fig_l = plt.figure(figsize=(6, 4))
                plt.plot(history["epoch"], history["train_loss"], label="train")
                plt.plot(history["epoch"], history["val_loss"], label="val")
                plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.tight_layout()
                fig_l.savefig(os.path.join(cfg.out_dir, "loss_curve.png"))
                plt.close(fig_l)

                # Accuracy curve
                fig_a = plt.figure(figsize=(6, 4))
                plt.plot(history["epoch"], history["train_acc"], label="train")
                plt.plot(history["epoch"], history["val_acc"], label="val")
                plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy Curve"); plt.legend(); plt.tight_layout()
                fig_a.savefig(os.path.join(cfg.out_dir, "accuracy_curve.png"))
                plt.close(fig_a)
            except Exception:
                pass

    cleanup_distributed()


if __name__ == "__main__":
    main()
