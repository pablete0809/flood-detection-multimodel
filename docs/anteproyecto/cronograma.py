"""Genera el diagrama de Gantt del anteproyecto (agosto 2026 - julio 2027).

Uso:
    .venv/bin/python docs/anteproyecto/cronograma.py

Salidas (junto a este script): cronograma.png (300 dpi), cronograma.pdf, cronograma.svg
Las fechas se expresan en meses desde el 1 de agosto de 2026 (0 = 1-ago, 0.5 = 15-ago).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent

MESES = ["Ago", "Sep", "Oct", "Nov", "Dic", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul"]

# (fase, color, [(tarea, inicio, fin), ...])
FASES = [
    ("1. Planificación y estado del arte", "#4C72B0", [
        ("Revisión bibliográfica (SAR, inundaciones, deep learning)", 0.0, 2.5),
        ("Definición de objetivos y anteproyecto", 2.0, 4.0),
    ]),
    ("2. Datos y preprocesado", "#55A868", [
        ("Selección de casos y descarga de datos (S1, S2, DEM, EMS)", 1.0, 3.0),
        ("Preprocesado SAR y generación de pseudo-etiquetas", 2.5, 4.5),
    ]),
    ("3. Desarrollo de modelos", "#C9853A", [
        ("Modelos clásicos (LogReg, RF, XGBoost, MLP)", 3.5, 5.5),
        ("Redes de segmentación (U-Net, Attention U-Net, DeepLabV3+)", 4.5, 7.0),
    ]),
    ("4. Experimentación y evaluación", "#C44E52", [
        ("Diseño experimental y validación espacial por bloques", 5.5, 7.0),
        ("Experimentos: supervisión débil frente a completa", 6.0, 8.0),
        ("Análisis de resultados y aplicación a Málaga", 7.5, 9.0),
    ]),
    ("5. Redacción y difusión", "#8172B2", [
        ("Redacción de la memoria", 5.0, 10.5),
        ("Redacción del artículo científico", 8.5, 10.0),
        ("Documentación y publicación del código", 9.0, 10.5),
        ("Revisión final con el tutor", 10.0, 11.0),
    ]),
    ("6. Defensa", "#5B8E9B", [
        ("Preparación de la presentación y defensa", 10.5, 12.0),
    ]),
]

HITOS = [
    ("H1  Entrega del anteproyecto", 3.9),
    ("H2  Pipeline de datos operativo", 4.5),
    ("H3  Resultados experimentales cerrados", 9.0),
    ("H4  Depósito de la memoria y el código", 11.0),
    ("H5  Defensa del TFG", 11.6),
]

TXT = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
SHADE = "#f4f4f4"


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    # Construir filas: (tipo, etiqueta, inicio, fin, color)
    filas = []
    for fase, color, tareas in FASES:
        ini = min(t[1] for t in tareas)
        fin = max(t[2] for t in tareas)
        filas.append(("fase", fase, ini, fin, color))
        for nombre, a, b in tareas:
            filas.append(("tarea", nombre, a, b, color))
    filas.append(("hitos", "Hitos", None, None, TXT))

    n = len(filas)
    fig, ax = plt.subplots(figsize=(13, 0.3 * n + 2.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(n + 1.1, -1.9)
    ax.axis("off")

    # Columna de etiquetas a la izquierda (en coordenadas de datos negativas)
    label_x = -0.15

    # Sombreado alterno de meses y rejilla
    for m in range(12):
        if m % 2 == 0:
            ax.add_patch(plt.Rectangle((m, -0.6), 1, n + 0.2, color=SHADE, zorder=0, lw=0))
        ax.plot([m, m], [-0.6, n - 0.4], color=GRID, lw=0.6, zorder=1)
        ax.text(m + 0.5, -0.95, MESES[m], ha="center", va="center", fontsize=9, color=TXT)
    ax.plot([12, 12], [-0.6, n - 0.4], color=GRID, lw=0.6, zorder=1)

    # Años
    ax.text(2.5, -1.55, "2026", ha="center", va="center", fontsize=10.5, weight="bold", color=TXT)
    ax.text(8.5, -1.55, "2027", ha="center", va="center", fontsize=10.5, weight="bold", color=TXT)
    ax.plot([5, 5], [-1.8, n - 0.4], color="#9a9a9a", lw=1.1, zorder=2)
    ax.plot([0, 12], [-0.6, -0.6], color="#9a9a9a", lw=0.8, zorder=2)

    for i, (tipo, nombre, a, b, color) in enumerate(filas):
        if tipo == "fase":
            ax.text(label_x, i, nombre, ha="right", va="center", fontsize=9.5,
                    weight="bold", color=color if nombre != "Hitos" else TXT)
            if a is not None:
                ax.add_patch(plt.Rectangle((a, i - 0.09), b - a, 0.18, color=color,
                                           alpha=0.55, zorder=3, lw=0))
        elif tipo == "tarea":
            ax.text(label_x, i, nombre, ha="right", va="center", fontsize=8.8, color=TXT)
            ax.add_patch(FancyBboxPatch((a + 0.04, i - 0.29), b - a - 0.08, 0.58,
                                        boxstyle="round,pad=0,rounding_size=0.12",
                                        fc=color, ec="none", zorder=3,
                                        mutation_aspect=0.5))
        else:  # fila única de hitos
            ax.text(label_x, i, nombre, ha="right", va="center", fontsize=9.5,
                    weight="bold", color=TXT)
            for nombre_h, x in HITOS:
                ax.plot([x, x], [i, -0.6], color=MUTED, lw=0.6, ls=(0, (2, 3)), zorder=2)
                ax.scatter([x], [i], marker="D", s=55, color=TXT, zorder=4)
                ax.text(x, i + 0.55, nombre_h.split()[0], ha="center", va="top",
                        fontsize=8.5, weight="bold", color=TXT)

    # Leyenda de hitos bajo el gráfico
    items = [f"{h.split(maxsplit=1)[0]}: {h.split(maxsplit=1)[1]}" for h, _ in HITOS]
    leyenda = "     ".join(items[:3]) + "\n" + "     ".join(items[3:])
    ax.text(6, n + 0.45, leyenda, ha="center", va="top", fontsize=8.6, color=TXT,
            linespacing=1.6)

    ax.set_title("Cronograma del Trabajo de Fin de Grado (agosto 2026 – julio 2027)",
                 fontsize=12.5, weight="bold", color=TXT, loc="center", pad=10)

    fig.subplots_adjust(left=0.36, right=0.985, top=0.93, bottom=0.03)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"cronograma.{ext}", facecolor="white", bbox_inches="tight", **kw)
    print("Guardado en", OUT)


if __name__ == "__main__":
    main()
