from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    data_root: str
    batch_size: int = 64
    val_split: float = 0.2
    num_workers: int = 2
    seed: int = 42
    image_size: int = 224


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    train = root / "train"
    test = root / "test"
    if train.exists() and test.exists():
        return root
    nested = root / "landmark_images"
    if (nested / "train").exists() and (nested / "test").exists():
        return nested
    raise FileNotFoundError(
        f"No se encontro estructura train/test en {root} ni {nested}."
    )


def build_dataloaders(
    cfg: DataConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    root = _resolve_data_root(cfg.data_root)
    train_dir = root / "train"
    test_dir = root / "test"

    base_train = datasets.ImageFolder(train_dir)
    eval_train = datasets.ImageFolder(train_dir, transform=_eval_transform(cfg.image_size))
    aug_train = datasets.ImageFolder(train_dir, transform=_train_transform(cfg.image_size))
    test_set = datasets.ImageFolder(test_dir, transform=_eval_transform(cfg.image_size))

    n_total = len(base_train)
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)

    train_set = Subset(aug_train, list(train_idx.indices))
    val_set = Subset(eval_train, list(val_idx.indices))

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, base_train.class_to_idx
