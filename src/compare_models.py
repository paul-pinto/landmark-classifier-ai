from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara resultados scratch vs transfer")
    parser.add_argument("--scratch-summary", type=str, default="outputs/scratch_run/summary.json")
    parser.add_argument("--transfer-summary", type=str, default="outputs/transfer_run/summary.json")
    parser.add_argument("--transfer2-summary", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/comparison")
    parser.add_argument("--scratch-threshold", type=float, default=0.40)
    parser.add_argument("--transfer-threshold", type=float, default=0.75)
    return parser.parse_args()


def to_row(label: str, summary: dict) -> dict:
    test_acc = float(summary.get("test_acc", 0.0))
    return {
        "model": label,
        "model_type": summary.get("model_type"),
        "backbone": summary.get("backbone"),
        "epochs": int(summary.get("epochs", 0)),
        "test_acc": test_acc,
        "test_acc_pct": round(test_acc * 100.0, 2),
        "test_loss": float(summary.get("test_loss", 0.0)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scratch = load_summary(Path(args.scratch_summary))
    transfer = load_summary(Path(args.transfer_summary))
    transfer2 = load_summary(Path(args.transfer2_summary)) if args.transfer2_summary else None

    rows = [to_row("scratch", scratch), to_row("transfer", transfer)]
    if transfer2 is not None:
        transfer2_label = f"transfer_{transfer2.get('backbone', 'alt')}"
        rows.append(to_row(transfer2_label, transfer2))

    with open(out_dir / "model_comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "model_type",
                "backbone",
                "epochs",
                "test_acc",
                "test_acc_pct",
                "test_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Comparación de Modelos",
        "",
        "| Modelo | Tipo | Backbone | Epochs | Test Accuracy | Test Loss |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['model']} | {row['model_type']} | {row['backbone']} | "
            f"{int(row['epochs'])} | {row['test_acc_pct']:.2f}% | {row['test_loss']:.4f} |"
        )

    beats = float(transfer.get("test_acc", 0.0)) > float(scratch.get("test_acc", 0.0))
    scratch_ok = float(scratch.get("test_acc", 0.0)) >= args.scratch_threshold
    transfer_ok = float(transfer.get("test_acc", 0.0)) >= args.transfer_threshold
    md_lines.append("")
    md_lines.append(f"- Scratch cumple umbral ({args.scratch_threshold*100:.0f}%): {'sí' if scratch_ok else 'no'}")
    md_lines.append(f"- Transfer cumple umbral ({args.transfer_threshold*100:.0f}%): {'sí' if transfer_ok else 'no'}")
    md_lines.append(
        f"- Transfer supera a scratch: {'sí' if beats else 'no'}"
    )
    if transfer2 is not None:
        best_transfer = max(
            [
                ("transfer", float(transfer.get("test_acc", 0.0))),
                (
                    f"transfer_{transfer2.get('backbone', 'alt')}",
                    float(transfer2.get("test_acc", 0.0)),
                ),
            ],
            key=lambda x: x[1],
        )
        md_lines.append(f"- Mejor variante transfer: {best_transfer[0]} ({best_transfer[1] * 100:.2f}%)")

    (out_dir / "model_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        labels = [row["model"] for row in rows]
        values = [row["test_acc_pct"] for row in rows]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=["#4c78a8", "#59a14f", "#f28e2b"][: len(rows)])
        plt.axhline(args.scratch_threshold * 100.0, color="#4c78a8", linestyle="--", linewidth=1.5, label="Umbral scratch")
        plt.axhline(args.transfer_threshold * 100.0, color="#59a14f", linestyle="--", linewidth=1.5, label="Umbral transfer")
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, val + 0.5, f"{val:.2f}%", ha="center", va="bottom")
        plt.ylim(0, 100)
        plt.ylabel("Test Accuracy (%)")
        plt.title("Comparación de Accuracy en Test")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "test_accuracy_comparison.png", dpi=140)
        plt.close()
    except Exception as exc:
        print(f"warning: no se pudo generar gráfico de comparación ({exc})")

    for row in rows:
        print(
            f"{row['model']:>20} | acc={row['test_acc_pct']:.2f}% | "
            f"loss={row['test_loss']:.4f} | backbone={row['backbone']}"
        )
    print(f"\nComparación guardada en: {out_dir}")


if __name__ == "__main__":
    main()
