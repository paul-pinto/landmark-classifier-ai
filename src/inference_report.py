from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from .predictor import predict_landmarks

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            yield p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera reporte de inferencia top-k para carpeta de imágenes")
    parser.add_argument("--images-dir", type=str, default="inference_images")
    parser.add_argument("--model-path", type=str, default="models/transfer_best.pt")
    parser.add_argument("--class-map", type=str, default="outputs/transfer_run/class_to_idx.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="outputs/inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(images_dir))
    if len(images) < 4:
        raise RuntimeError(
            f"Se requieren al menos 4 imágenes en {images_dir}. Encontradas: {len(images)}"
        )

    csv_path = out_dir / "inference_results.csv"
    md_path = out_dir / "inference_results.md"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "rank", "label", "probability"])
        for img in images:
            preds = predict_landmarks(
                img_path=img,
                model_path=args.model_path,
                class_map_path=args.class_map,
                k=args.k,
            )
            for rank, (label, prob) in enumerate(preds, start=1):
                writer.writerow([img.name, rank, label, f"{prob:.6f}"])

    lines = [
        "# Resultados de Inferencia (Top-k)",
        "",
        f"- Modelo: `{args.model_path}`",
        f"- Class map: `{args.class_map}`",
        f"- Imágenes evaluadas: {len(images)}",
        f"- Top-k: {args.k}",
        "",
    ]
    for img in images:
        lines.append(f"## {img.name}")
        preds = predict_landmarks(
            img_path=img,
            model_path=args.model_path,
            class_map_path=args.class_map,
            k=args.k,
        )
        for rank, (label, prob) in enumerate(preds, start=1):
            lines.append(f"- {rank}. {label}: {prob:.4f}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()
