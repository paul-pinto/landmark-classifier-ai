from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _pass_fail(ok: bool) -> str:
    return "OK" if ok else "PENDIENTE"


def build_report(
    scratch_summary: dict[str, Any] | None,
    transfer_summary: dict[str, Any] | None,
) -> str:
    scratch_acc = scratch_summary.get("test_acc") if scratch_summary else None
    transfer_acc = transfer_summary.get("test_acc") if transfer_summary else None
    scratch_epochs = int(scratch_summary.get("epochs", 0)) if scratch_summary else 0
    transfer_epochs = int(transfer_summary.get("epochs", 0)) if transfer_summary else 0

    scratch_ok = (scratch_acc is not None) and (scratch_acc >= 0.40)
    transfer_ok = (transfer_acc is not None) and (transfer_acc >= 0.75)
    scratch_epochs_ok = scratch_epochs >= 30
    transfer_epochs_ok = transfer_epochs >= 1
    beats_scratch = (
        transfer_acc is not None and scratch_acc is not None and transfer_acc > scratch_acc
    )

    lines: list[str] = []
    lines.append("# Estado del Proyecto")
    lines.append("")
    lines.append("## Métricas clave")
    lines.append("")
    lines.append("| Modelo | Test Accuracy | Umbral | Estado |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| Scratch | {_fmt_pct(scratch_acc)} | 40.00% | {_pass_fail(scratch_ok)} |")
    lines.append(f"| Transfer | {_fmt_pct(transfer_acc)} | 75.00% | {_pass_fail(transfer_ok)} |")
    lines.append("")
    lines.append("## Chequeo de rúbrica (automático)")
    lines.append("")
    lines.append(
        f"- Scratch entrenado >=30 épocas: {_pass_fail(scratch_epochs_ok)}"
    )
    lines.append(
        f"- Fase 2 (scratch `>=40%`): {_pass_fail(scratch_ok)}"
    )
    lines.append(
        f"- Transfer entrenado: {_pass_fail(transfer_epochs_ok)}"
    )
    lines.append(
        f"- Fase 3 (transfer `>=75%`): {_pass_fail(transfer_ok)}"
    )
    lines.append(
        f"- Transfer supera scratch: {_pass_fail(beats_scratch)}"
    )
    lines.append("")
    lines.append("## Pendientes manuales")
    lines.append("")
    lines.append("- Validar notebook ejecutado con salidas visibles.")
    lines.append("- Incluir 4 imágenes propias con predicción top-k.")
    lines.append("- Completar análisis escrito (fortalezas, debilidades, mejoras).")
    lines.append("- Subir video de 3 a 5 minutos con enlace.")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera estado de cumplimiento del proyecto")
    parser.add_argument("--scratch-summary", type=str, default="outputs/scratch_run/summary.json")
    parser.add_argument("--transfer-summary", type=str, default="outputs/transfer_run/summary.json")
    parser.add_argument("--out", type=str, default="outputs/project_status.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scratch_path = Path(args.scratch_summary)
    transfer_path = Path(args.transfer_summary)

    scratch_summary = _load_json(scratch_path) if scratch_path.exists() else None
    transfer_summary = _load_json(transfer_path) if transfer_path.exists() else None

    report = build_report(scratch_summary, transfer_summary)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReporte guardado en: {out_path}")


if __name__ == "__main__":
    main()
