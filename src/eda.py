from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import datasets

from .data import _resolve_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA para dataset de landmarks")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/eda")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root = _resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.ImageFolder(root / "train")
    test_ds = datasets.ImageFolder(root / "test")

    class_counts: dict[str, int] = {}
    for _, class_idx in train_ds.samples:
        class_name = train_ds.classes[class_idx]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    with open(out_dir / "class_distribution_train.json", "w", encoding="utf-8") as f:
        json.dump(class_counts, f, ensure_ascii=False, indent=2)

    sorted_items = sorted(class_counts.items(), key=lambda x: x[0])
    labels = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    plt.figure(figsize=(16, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.ylabel("Cantidad de imágenes (train)")
    plt.title("Distribución de clases - Train")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution_train.png", dpi=140)
    plt.close()

    num_samples = min(args.num_samples, len(train_ds))
    indices = random.sample(range(len(train_ds)), num_samples)

    cols = 3
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(12, 4 * rows))
    for i, idx in enumerate(indices, start=1):
        img, cls_idx = train_ds[idx]
        cls_name = train_ds.classes[cls_idx]
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(cls_name)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_images_train.png", dpi=140)
    plt.close()

    summary = {
        "train_images": len(train_ds),
        "test_images": len(test_ds),
        "num_classes": len(train_ds.classes),
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"EDA guardado en: {out_dir}")


if __name__ == "__main__":
    main()
