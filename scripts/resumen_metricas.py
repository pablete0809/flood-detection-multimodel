"""
Resume uno o varios manifest_compare.json en tablas Markdown (para el README o la memoria).

Uso:
    python scripts/resumen_metricas.py outputs/<run>/compare_pseudo/metadata/manifest_compare.json
    python scripts/resumen_metricas.py --task flood m1.json m2.json   # una tabla por manifiesto

Las métricas son las "overall" de cada modelo (media entre ventanas de evento).
Manifiestos antiguos sin umbral de validación muestran "—" en esas columnas.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COLUMNS = [
    ("auc_roc", "AUC-ROC"),
    ("pr_auc", "PR-AUC"),
    ("f1", "F1"),
    ("iou", "IoU"),
    ("threshold", "Umbral"),
    ("f1_at_05", "F1@0.5"),
    ("brier", "Brier"),
    ("ece", "ECE"),
]


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}"


def table(manifest: dict, task: str) -> str:
    rows = []
    for model, res in manifest["results"].items():
        overall = res.get(task, {}).get("overall")
        if overall:
            rows.append((model, overall))
    if not rows:
        return f"_Sin resultados para la tarea '{task}'._\n"

    lines = [
        "| Modelo | " + " | ".join(name for _, name in COLUMNS) + " |",
        "|---|" + "---|" * len(COLUMNS),
    ]
    for model, overall in rows:
        lines.append(
            f"| {model} | " + " | ".join(_fmt(overall.get(k)) for k, _ in COLUMNS) + " |"
        )
    n_test = rows[0][1].get("n_test")
    n_pos = rows[0][1].get("n_test_pos")
    if n_test:
        lines.append("")
        miles = lambda n: f"{int(n):,}".replace(",", ".")  # noqa: E731
        lines.append(
            f"_Píxeles de test: {miles(n_test)} ({miles(n_pos)} positivos, {n_pos / n_test:.1%})._"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--task", choices=["water", "flood", "both"], default="both")
    args = parser.parse_args()

    tasks = ["water", "flood"] if args.task == "both" else [args.task]
    for path in args.manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        protocol = manifest.get("protocol", {})
        print(f"## {path}")
        print(f"train_on = {protocol.get('train_on', 'desconocido (manifiesto anterior al protocolo)')}\n")
        for task in tasks:
            print(f"### {task}\n")
            print(table(manifest, task))


if __name__ == "__main__":
    main()
