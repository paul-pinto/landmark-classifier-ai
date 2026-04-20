from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD


def _build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _load_idx_to_class(class_map_path: str | Path) -> dict[int, str]:
    class_to_idx = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
    return {idx: cls for cls, idx in class_to_idx.items()}


def predict_landmarks(
    img_path: str | Path,
    model_path: str | Path,
    class_map_path: str | Path,
    k: int = 5,
    image_size: int = 224,
) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    idx_to_class = _load_idx_to_class(class_map_path)
    transform = _build_transform(image_size=image_size)

    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_idxs = top_idxs.squeeze(0).cpu().tolist()

    results = []
    for idx, prob in zip(top_idxs, top_probs):
        cls = idx_to_class.get(idx, f"class_{idx}")
        results.append((cls, float(prob)))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict top-k landmarks")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--class-map", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preds = predict_landmarks(
        img_path=args.image,
        model_path=args.model_path,
        class_map_path=args.class_map,
        k=args.k,
        image_size=args.image_size,
    )
    for rank, (label, prob) in enumerate(preds, start=1):
        print(f"{rank}. {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
