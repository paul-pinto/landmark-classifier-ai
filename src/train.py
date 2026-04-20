from __future__ import annotations

import argparse
import csv
import contextlib
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import DataConfig, build_dataloaders
from .model import build_model


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.set_grad_enabled(is_train):
            with autocast_ctx:
                logits = model(xb)
                loss = criterion(logits, yb)
            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())
        total_samples += int(yb.numel())

    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": total_correct / max(total_samples, 1),
    }


def plot_curves(history: List[Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: no se pudieron generar curvas con matplotlib ({exc})")
        return
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=140)
    plt.close()


def export_torchscript(model: nn.Module, export_path: Path, image_size: int, device: torch.device) -> None:
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    traced = torch.jit.trace(model, dummy)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(export_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train landmark classifier")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--model-type",
        choices=["scratch", "scratch_resnet18", "transfer"],
        required=True,
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-cosine-scheduler", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="outputs/run")
    parser.add_argument("--export-path", type=str, default="models/best_model.pt")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    return parser.parse_args()


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    scaler: torch.cuda.amp.GradScaler | None,
    history: List[Dict[str, float]],
    best_val_loss: float,
    best_state: Dict[str, torch.Tensor] | None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_state": best_state,
        },
        checkpoint_path,
    )


def save_epoch_snapshot(
    checkpoint_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        },
        epoch_path,
    )
    return epoch_path


def maybe_load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, List[Dict[str, float]], float, Dict[str, torch.Tensor] | None]:
    if not checkpoint_path.exists():
        return 0, [], float("inf"), None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    sched_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    start_epoch = int(checkpoint.get("epoch", 0))
    history = checkpoint.get("history", [])
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_state = checkpoint.get("best_state")
    return start_epoch, history, best_val_loss, best_state


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("cudnn_benchmark=enabled")
    if device.type == "cpu":
        torch.backends.mkldnn.enabled = False
        print("mkldnn=disabled (cpu stability mode)")
    use_amp = args.amp if args.amp is not None else (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None
    print(f"amp={use_amp}")

    cfg = DataConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
    )

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(cfg)
    num_classes = len(class_to_idx)
    print(f"classes={num_classes}, train_batches={len(train_loader)}")

    with open(out_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    model = build_model(args.model_type, num_classes=num_classes, backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_cosine_scheduler else None
    )

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else out_dir / "checkpoint_last.pth"
    )
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else out_dir / "checkpoints"
    )
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every debe ser >= 1")

    if args.resume:
        start_epoch, history, best_val_loss, best_state = maybe_load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        print(f"resume={args.resume}, start_epoch={start_epoch}")
    else:
        start_epoch, history, best_val_loss, best_state = 0, [], float("inf"), None

    if history:
        best_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
    else:
        best_epoch = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            scaler=None,
            use_amp=use_amp,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step()

        save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_val_loss=best_val_loss,
            best_state=best_state,
        )
        if (epoch % args.checkpoint_every) == 0:
            snapshot_path = save_epoch_snapshot(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"checkpoint_epoch={snapshot_path}")

    if best_state is None:
        raise RuntimeError("No se encontro mejor estado del modelo.")

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_weights.pth")

    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        None,
        device,
        scaler=None,
        use_amp=use_amp,
    )
    print(f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['acc']:.4f}")

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "model_type": args.model_type,
                "backbone": args.backbone if args.model_type == "transfer" else None,
                "epochs": args.epochs,
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "num_classes": num_classes,
                "best_epoch": best_epoch,
                "checkpoint_last": str(checkpoint_path),
                "checkpoint_dir": str(checkpoint_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_curves(history, out_dir)
    export_torchscript(model, Path(args.export_path), image_size=args.image_size, device=device)
    print(f"exported_torchscript={args.export_path}")


if __name__ == "__main__":
    main()
