from __future__ import annotations

"""Publication-quality figures for the model comparison framework.

Figures are saved to outputs/{run_name}/figures/ as both PNG (300 dpi) and PDF.

Public entry point:
    run_visualize_compare(cfg, results, paths, targets_meta)

Individual figure functions can also be called directly.

Figures generated:
    A. sar_context_{key}.png       — SAR antes/después/cambio con ground truth contour
    B. model_predictions_{key}.png — grid ground truth + predicciones de cada modelo
    C. ensemble_analysis_{key}.png — acuerdo, incertidumbre, mejor/peor modelo
    + ROC, PR, calibration, metrics bar, feature importance (sin cambios)
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401

from floodprob.compare_metrics import get_roc_curve, get_pr_curve, get_calibration_curve
from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import reference_labels
from floodprob.splits import SPLIT_TEST, split_for_cfg

LOGGER = logging.getLogger(__name__)

# ── Colour palette per model family ──────────────────────────────────────────
_MODEL_COLORS = {
    "threshold_change": "#555555",
    "logreg":           "#2196F3",
    "random_forest":    "#4CAF50",
    "xgboost":          "#FF9800",
    "mlp":              "#9C27B0",
    "unet":             "#F44336",
    "attention_unet":   "#E91E63",
    "deeplabv3plus":    "#00BCD4",
}
_DEFAULT_COLOR = "#888888"

_ERROR_CMAP = "hot_r"


# ── Save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s and %s", path, pdf_path)


def _model_color(name: str) -> str:
    return _MODEL_COLORS.get(name, _DEFAULT_COLOR)


# ── North arrow + scale bar utilities ────────────────────────────────────────

def _add_north_arrow(ax: plt.Axes, x: float = 0.05, y: float = 0.95) -> None:
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - 0.07),
        xycoords="axes fraction", textcoords="axes fraction",
        fontsize=8, ha="center", va="center", weight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.5),
    )


def _add_scale_bar(
    ax: plt.Axes,
    da: xr.DataArray,
    length_m: float = 5000,
    x0_frac: float = 0.05,
    y0_frac: float = 0.06,
) -> None:
    """Draw a simple scale bar. Assumes the DataArray's CRS is metric."""
    try:
        res = float(abs(da.x[1] - da.x[0]))
        if res == 0:
            return
        n_pixels = length_m / res
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_start = xlim[0] + x0_frac * (xlim[1] - xlim[0])
        y_pos = ylim[0] + y0_frac * (ylim[1] - ylim[0])
        ax.hlines(y_pos, x_start, x_start + n_pixels * res, colors="k", linewidths=2)
        ax.text(
            x_start + n_pixels * res / 2, y_pos + 0.015 * (ylim[1] - ylim[0]),
            f"{length_m / 1000:.0f} km", ha="center", va="bottom", fontsize=7,
        )
    except Exception:  # noqa: BLE001
        pass


# ── Shared raster loader ──────────────────────────────────────────────────────

def _load_raster(path: str | None) -> np.ndarray | None:
    """Load a GeoTIFF as a 2-D float32 array. Returns None on any failure."""
    import rioxarray as rxr
    if not path or not Path(path).exists():
        return None
    try:
        return rxr.open_rasterio(path, masked=True).squeeze(drop=True).values.astype(np.float32)
    except Exception:
        LOGGER.error("Failed to load raster: %s", path)
        return None


def _load_reference_label(labels_path: str | None, task: str) -> np.ndarray | None:
    """Etiquetas de referencia de la ventana: ground truth si existe, si no pseudo-etiquetas."""
    if not labels_path or not Path(labels_path).exists():
        return None
    try:
        lab_ds = xr.open_dataset(Path(labels_path))
        return reference_labels(lab_ds, task)[0].values.astype(np.float32)
    except Exception:  # noqa: BLE001
        LOGGER.warning("No se pudieron cargar etiquetas de %s desde %s", task, labels_path)
        return None


def _get_per_window_metric(results: dict, model: str, task: str, key: str, metric: str) -> float:
    per_w = results[model].get(task, {}).get("per_window", [])
    w = next((wp for wp in per_w if wp["key"] == key), None)
    if w and w.get("metrics"):
        return float(w["metrics"].get(metric, float("nan")))
    return float("nan")


# ── A: SAR context (antes / después / cambio) ────────────────────────────────

def plot_sar_context(
    target: dict,
    key: str,
    figures_dir: Path,
) -> None:
    """Figura A: VV y VH antes/después del evento + mapa de cambio.

    Reconstruye el baseline como vv_db_base = vv_db - delta_vv (idéntico
    a lo que se calculó en preprocess, no se guarda en features_*.nc).
    Superpone el contorno de flood_label en rojo si está disponible.
    """
    features_path = target.get("features_path")
    labels_path = target.get("labels_path")

    if not features_path or not Path(features_path).exists():
        LOGGER.error("plot_sar_context: features_path no encontrado para key=%s", key)
        return

    feat_ds = xr.open_dataset(Path(features_path))

    required = {"vv_db", "delta_vv"}
    if not required.issubset(set(feat_ds.data_vars)):
        LOGGER.error("plot_sar_context: faltan variables SAR en features para key=%s", key)
        return

    vv_after  = feat_ds["vv_db"].values.astype(np.float32)
    delta_vv  = feat_ds["delta_vv"].values.astype(np.float32)
    vv_before = vv_after - delta_vv

    has_vh = "vh_db" in feat_ds and "delta_vh" in feat_ds
    if has_vh:
        vh_after  = feat_ds["vh_db"].values.astype(np.float32)
        delta_vh  = feat_ds["delta_vh"].values.astype(np.float32)
        vh_before = vh_after - delta_vh

    # Load flood reference label for contour overlay
    flood_label = _load_reference_label(labels_path, "flood")

    n_rows = 2 if has_vh else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.8 * n_rows), squeeze=False)
    fig.suptitle(f"Contexto SAR — ventana {key}", fontsize=13, fontweight="bold")

    for j, title in enumerate(["Antes (baseline)", "Después (evento)", "Cambio (Δ)"]):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    def _pct_clip(arr: np.ndarray) -> tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return 0.0, 1.0
        return float(np.percentile(finite, 2)), float(np.percentile(finite, 98))

    def _plot_gray(ax: plt.Axes, arr: np.ndarray, ylabel: str) -> None:
        vmin, vmax = _pct_clip(arr)
        im = ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_ylabel(ylabel, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")
        ax.set_xticks([]); ax.set_yticks([])

    def _plot_diverging(ax: plt.Axes, arr: np.ndarray, ylabel: str) -> None:
        finite = arr[np.isfinite(arr)]
        vabs = max(float(np.percentile(np.abs(finite), 99)), 0.1) if len(finite) else 1.0
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="upper")
        ax.set_ylabel(ylabel, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ dB")
        ax.set_xticks([]); ax.set_yticks([])

    def _add_flood_contour(ax: plt.Axes) -> None:
        if flood_label is None:
            return
        valid = (flood_label >= 0) & np.isfinite(flood_label)
        if valid.any() and (flood_label[valid] > 0).any():
            safe = np.where(flood_label >= 0, flood_label, 0.0)
            ax.contour(safe, levels=[0.5], colors=["red"], linewidths=1.5)

    # Row 0: VV
    _plot_gray(axes[0, 0], vv_before, "VV (dB)")
    _plot_gray(axes[0, 1], vv_after,  "VV (dB)")
    _plot_diverging(axes[0, 2], delta_vv, "Δ VV (dB)")
    for j in range(3):
        _add_flood_contour(axes[0, j])

    # Row 1: VH
    if has_vh:
        _plot_gray(axes[1, 0], vh_before, "VH (dB)")
        _plot_gray(axes[1, 1], vh_after,  "VH (dB)")
        _plot_diverging(axes[1, 2], delta_vh, "Δ VH (dB)")
        for j in range(3):
            _add_flood_contour(axes[1, j])

    # North arrow + scale bar (using xarray DataArray for pixel size)
    try:
        da_ref = feat_ds["vv_db"]
        for i in range(n_rows):
            _add_north_arrow(axes[i, 0])
        _add_scale_bar(axes[0, 0], da_ref)
    except Exception:  # noqa: BLE001
        pass

    fig.tight_layout()
    _save(fig, figures_dir / f"sar_context_{key}.png")


# ── B: Model predictions grid ────────────────────────────────────────────────

def plot_model_predictions(
    results: dict[str, Any],
    target: dict,
    key: str,
    figures_dir: Path,
) -> None:
    """Figura B: Ground truth + predicciones de todos los modelos.

    Fila 0: P(agua) — un panel por water model, escala fija [0,1], Blues.
    Fila 1: P(inundación|agua) — un panel por flood model, escala fija [0,1], Reds.
    Columna 0 de cada fila: ground truth correspondiente.
    Contorno blanco al umbral 0.5 sobre cada predicción.
    Métricas AUC / F1 / IoU en el título de cada panel.
    """
    import rioxarray as rxr  # noqa: F401 (ensures plugin registered)

    water_models = [m for m in results if results[m].get("water", {}).get("per_window")]
    flood_models = [m for m in results if results[m].get("flood", {}).get("per_window")]

    if not water_models and not flood_models:
        LOGGER.warning("plot_model_predictions: sin modelos con datos per_window para key=%s", key)
        return

    n_water = len(water_models)
    n_flood = len(flood_models)
    n_cols = 1 + max(n_water, n_flood)

    # Load reference labels
    labels_path = target.get("labels_path")
    water_label = _load_reference_label(labels_path, "water")
    flood_label = _load_reference_label(labels_path, "flood")

    fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 7.5), squeeze=False)
    fig.suptitle(f"Predicciones por modelo — ventana {key}", fontsize=13, fontweight="bold")

    axes[0, 0].set_ylabel("P(agua)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("P(inund.|agua)", fontsize=10, fontweight="bold")

    # ── Column 0: ground truth ────────────────────────────────────────────────
    for row, label_arr, cmap, row_label in [
        (0, water_label, "Blues", "Etiqueta agua"),
        (1, flood_label, "Reds",  "Etiqueta inund."),
    ]:
        ax = axes[row, 0]
        ax.set_title(row_label, fontsize=9, fontweight="bold")
        if label_arr is not None:
            disp = np.where(label_arr >= 0, label_arr, np.nan)
            ax.imshow(disp, cmap=cmap, vmin=0, vmax=1, origin="upper")
        else:
            ax.text(0.5, 0.5, "Sin etiqueta", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_xticks([]); ax.set_yticks([])

    # ── Row 0: water model predictions ───────────────────────────────────────
    water_mappable = None
    for ci, m in enumerate(water_models):
        pw_path = next(
            (wp["p_agua"] for wp in results[m]["water"]["per_window"] if wp["key"] == key),
            None,
        )
        arr = _load_raster(pw_path)
        auc = _get_per_window_metric(results, m, "water", key, "auc_roc")

        ax = axes[0, ci + 1]
        title = m
        if np.isfinite(auc):
            title += f"\nAUC={auc:.3f}"
        ax.set_title(title, fontsize=8, fontweight="bold", color=_model_color(m))

        if arr is not None:
            arr_c = np.clip(arr, 0, 1)
            im = ax.imshow(arr_c, cmap="Blues", vmin=0, vmax=1, origin="upper")
            water_mappable = im
            ax.contour(arr_c, levels=[0.5], colors=["white"], linewidths=1.2)
            if flood_label is not None:
                safe = np.where(flood_label >= 0, flood_label, 0.0)
                ax.contour(safe, levels=[0.5], colors=["red"], linewidths=1.0,
                           linestyles=["--"])
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])

    for ci in range(n_water, n_cols - 1):
        axes[0, ci + 1].set_visible(False)

    # ── Row 1: flood model predictions ───────────────────────────────────────
    flood_mappable = None
    for ci, m in enumerate(flood_models):
        pf_path = next(
            (wp["p_inundacion_dado_agua"]
             for wp in results[m]["flood"]["per_window"] if wp["key"] == key),
            None,
        )
        arr = _load_raster(pf_path)
        f1  = _get_per_window_metric(results, m, "flood", key, "f1")
        iou = _get_per_window_metric(results, m, "flood", key, "iou")

        ax = axes[1, ci + 1]
        parts = [m]
        if np.isfinite(f1):
            parts.append(f"F1={f1:.3f}")
        if np.isfinite(iou):
            parts.append(f"IoU={iou:.3f}")
        ax.set_title("\n".join(parts[:1]) + ("\n" + "  ".join(parts[1:]) if len(parts) > 1 else ""),
                     fontsize=8, fontweight="bold", color=_model_color(m))

        if arr is not None:
            arr_c = np.clip(arr, 0, 1)
            im = ax.imshow(arr_c, cmap="Reds", vmin=0, vmax=1, origin="upper")
            flood_mappable = im
            ax.contour(arr_c, levels=[0.5], colors=["white"], linewidths=1.2)
            if flood_label is not None:
                safe = np.where(flood_label >= 0, flood_label, 0.0)
                ax.contour(safe, levels=[0.5], colors=["black"], linewidths=1.0,
                           linestyles=["--"])
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])

    for ci in range(n_flood, n_cols - 1):
        axes[1, ci + 1].set_visible(False)

    # ── Shared colorbars per row ──────────────────────────────────────────────
    if water_mappable is not None:
        fig.colorbar(water_mappable, ax=axes[0, :].tolist(),
                     shrink=0.55, pad=0.01, label="P(agua)  [0 – 1]")
    if flood_mappable is not None:
        fig.colorbar(flood_mappable, ax=axes[1, :].tolist(),
                     shrink=0.55, pad=0.01, label="P(inund.|agua)  [0 – 1]")

    # North arrow + scale bar on first real data column
    try:
        da_ref = xr.open_dataset(Path(target["features_path"]))["vv_db"]
        _add_north_arrow(axes[0, 1])
        _add_scale_bar(axes[0, 1], da_ref)
    except Exception:  # noqa: BLE001
        pass

    # Legend for contours
    legend_handles = [
        mpatches.Patch(color="white", label="Umbral 0.5 (modelo)"),
        mpatches.Patch(color="red",   label="GT inundación"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    _save(fig, figures_dir / f"model_predictions_{key}.png")


# ── C: Ensemble analysis ──────────────────────────────────────────────────────

def plot_ensemble_analysis(
    results: dict[str, Any],
    target: dict,
    key: str,
    figures_dir: Path,
) -> None:
    """Figura C: Acuerdo entre modelos, incertidumbre, mejor y peor modelo.

    Panel 1 — Agreement: fracción de flood models con P > 0.5.
    Panel 2 — Uncertainty: desviación típica pixel a pixel entre modelos.
    Panel 3 — Mejor modelo (mayor F1).
    Panel 4 — Peor modelo (menor F1).
    Contorno blanco = ground truth flood si disponible.
    """
    import rioxarray as rxr  # noqa: F401

    flood_models = [m for m in results if results[m].get("flood", {}).get("per_window")]
    if len(flood_models) < 2:
        LOGGER.info(
            "plot_ensemble_analysis: menos de 2 flood models para key=%s, omitiendo", key
        )
        return

    # Load flood reference label for contour
    flood_label = _load_reference_label(target.get("labels_path"), "flood")

    # Load all flood predictions for this key
    pred_arrays: dict[str, np.ndarray] = {}
    for m in flood_models:
        pf_path = next(
            (wp["p_inundacion_dado_agua"]
             for wp in results[m]["flood"]["per_window"] if wp["key"] == key),
            None,
        )
        arr = _load_raster(pf_path)
        if arr is not None:
            pred_arrays[m] = np.clip(arr, 0, 1)

    if len(pred_arrays) < 2:
        LOGGER.warning(
            "plot_ensemble_analysis: no se cargaron suficientes rasters para key=%s", key
        )
        return

    stack     = np.stack(list(pred_arrays.values()), axis=0)  # (N, H, W)
    agreement = (stack > 0.5).mean(axis=0).astype(np.float32)
    uncertainty = stack.std(axis=0).astype(np.float32)

    # Rank models by F1
    f1_scores = {
        m: _get_per_window_metric(results, m, "flood", key, "f1")
        for m in pred_arrays
    }
    valid_f1 = {m: v for m, v in f1_scores.items() if np.isfinite(v)}
    if valid_f1:
        best_model  = max(valid_f1, key=lambda m: valid_f1[m])
        worst_model = min(valid_f1, key=lambda m: valid_f1[m])
    else:
        models_list = list(pred_arrays.keys())
        best_model, worst_model = models_list[0], models_list[-1]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    fig.suptitle(f"Análisis de ensemble — ventana {key}", fontsize=13, fontweight="bold")

    def _add_gt_contour(ax: plt.Axes) -> None:
        if flood_label is None:
            return
        valid = (flood_label >= 0) & np.isfinite(flood_label)
        if valid.any() and (flood_label[valid] > 0).any():
            safe = np.where(flood_label >= 0, flood_label, 0.0)
            ax.contour(safe, levels=[0.5], colors=["white"], linewidths=2.0)

    # Panel 1: Agreement
    ax = axes[0]
    im = ax.imshow(agreement, cmap="YlOrRd", vmin=0, vmax=1, origin="upper")
    ax.set_title(
        f"Acuerdo entre modelos\n({len(pred_arrays)} flood models)", fontsize=10
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fracción modelos > 0.5")
    _add_gt_contour(ax)
    _add_north_arrow(ax)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 2: Uncertainty
    ax = axes[1]
    im = ax.imshow(uncertainty, cmap="viridis", vmin=0, vmax=0.5, origin="upper")
    ax.set_title("Incertidumbre\n(σ entre modelos)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Desv. típica")
    _add_gt_contour(ax)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 3: Best model
    ax = axes[2]
    best_f1 = f1_scores.get(best_model, float("nan"))
    title3 = f"Mejor: {best_model}"
    if np.isfinite(best_f1):
        title3 += f"\nF1={best_f1:.3f}"
    im = ax.imshow(pred_arrays[best_model], cmap="Reds", vmin=0, vmax=1, origin="upper")
    ax.set_title(title3, fontsize=10, color=_model_color(best_model), fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(inund.|agua)")
    ax.contour(pred_arrays[best_model], levels=[0.5], colors=["white"], linewidths=1.2)
    _add_gt_contour(ax)
    ax.set_xticks([]); ax.set_yticks([])

    # Panel 4: Worst model
    ax = axes[3]
    worst_f1 = f1_scores.get(worst_model, float("nan"))
    title4 = f"Peor: {worst_model}"
    if np.isfinite(worst_f1):
        title4 += f"\nF1={worst_f1:.3f}"
    im = ax.imshow(pred_arrays[worst_model], cmap="Reds", vmin=0, vmax=1, origin="upper")
    ax.set_title(title4, fontsize=10, color=_model_color(worst_model), fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(inund.|agua)")
    ax.contour(pred_arrays[worst_model], levels=[0.5], colors=["white"], linewidths=1.2)
    _add_gt_contour(ax)
    ax.set_xticks([]); ax.set_yticks([])

    # Legend: white contour = GT
    if flood_label is not None:
        fig.legend(
            handles=[mpatches.Patch(color="white", label="Ground truth (contorno)")],
            loc="lower center", fontsize=8, framealpha=0.8, bbox_to_anchor=(0.5, -0.03),
        )

    fig.tight_layout()
    _save(fig, figures_dir / f"ensemble_analysis_{key}.png")


# ── External label loader (raster o vector) ───────────────────────────────────

def _load_external_label(
    ext_path: Path,
    results: dict[str, Any],
) -> np.ndarray | None:
    """Carga una etiqueta externa como array binario (H, W) uint8.

    Soporta:
    - GeoTIFF / raster: lee directamente con rioxarray.
    - GeoJSON / Shapefile (vector): rasteriza sobre la cuadrícula de la primera
      predicción disponible en results, usando geopandas + rasterio.features.
    """
    suffix = ext_path.suffix.lower()
    vector_exts = {".geojson", ".json", ".gpkg", ".shp"}

    if suffix not in vector_exts:
        # Raster path
        import rioxarray as rxr
        da = rxr.open_rasterio(str(ext_path), masked=True).squeeze(drop=True)
        return (da.values > 0).astype(np.uint8)

    # ── Vector path: rasterizar sobre la cuadrícula de referencia ────────────
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize as rio_rasterize

    # Find a reference prediction raster for shape/transform/CRS
    ref_raster: str | None = None
    for m in results:
        for task in ("water", "flood"):
            for wp in results[m].get(task, {}).get("per_window", []):
                candidate = wp.get("p_agua") or wp.get("p_inundacion_dado_agua")
                if candidate and Path(candidate).exists():
                    ref_raster = candidate
                    break
            if ref_raster:
                break
        if ref_raster:
            break

    if ref_raster is None:
        LOGGER.error("_load_external_label: no se encontró raster de referencia para rasterizar el vector")
        return None

    gdf = gpd.read_file(ext_path)
    with rasterio.open(ref_raster) as src:
        ref_crs   = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)

    if gdf.crs is None:
        LOGGER.warning("_load_external_label: GeoJSON sin CRS, asumiendo EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(ref_crs)

    valid_geoms = [
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]
    if not valid_geoms:
        LOGGER.warning("_load_external_label: el vector no contiene geometrías válidas")
        return np.zeros(ref_shape, dtype=np.uint8)

    arr = rio_rasterize(
        valid_geoms,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=0,
        dtype=np.uint8,
    )
    LOGGER.info(
        "_load_external_label: rasterizado %s → %d píxeles positivos de %d totales",
        ext_path.name, int(arr.sum()), arr.size,
    )
    return arr


# ── Curve data collector ──────────────────────────────────────────────────────

def _collect_labels_probs(
    results: dict[str, Any],
    task: str,
    label_key: str,
    label_arr_map: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns {model_name: (y_true_concat, y_prob_concat)} across all windows."""
    import rioxarray as rxr
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m, res in results.items():
        task_res = res.get(task, {})
        per_window = task_res.get("per_window", [])
        if not per_window:
            continue
        ys, ps = [], []
        for wp in per_window:
            k = wp["key"]
            path = wp.get(label_key)
            if not path or not Path(path).exists():
                continue
            if k not in label_arr_map:
                continue
            try:
                p_arr = rxr.open_rasterio(path, masked=True).squeeze(drop=True).values.astype(
                    np.float32
                )
            except Exception:  # noqa: BLE001
                continue
            y_arr = label_arr_map[k]
            mask = (y_arr >= 0) & np.isfinite(p_arr)
            if mask.sum() == 0:
                continue
            ys.append(y_arr[mask].astype(np.uint8))
            ps.append(np.clip(p_arr[mask], 0, 1))
        if ys:
            out[m] = (np.concatenate(ys), np.concatenate(ps))
    return out


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    figures_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")
    for m, (y, p) in model_data.items():
        fpr, tpr, auc = get_roc_curve(y, p)
        ax.plot(fpr, tpr, color=_model_color(m), lw=1.8, label=f"{m}  (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, figures_dir / f"{stem}.png")


# ── PR curves ─────────────────────────────────────────────────────────────────

def plot_pr_curves(
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    figures_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, (y, p) in model_data.items():
        prec, rec, auc = get_pr_curve(y, p)
        ax.plot(rec, prec, color=_model_color(m), lw=1.8, label=f"{m}  (AP={auc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, figures_dir / f"{stem}.png")


# ── Metric bar chart ──────────────────────────────────────────────────────────

def plot_metrics_bar(
    results: dict[str, Any],
    task: str,
    figures_dir: Path,
    stem: str,
) -> None:
    metrics_keys = ["auc_roc", "f1", "iou"]
    label_names  = ["AUC-ROC", "F1 (umbral val.)", "IoU (umbral val.)"]
    models = [m for m in results if results[m].get(task, {}).get("overall")]
    if not models:
        LOGGER.warning("No %s metrics to plot bar chart.", task)
        return

    x = np.arange(len(metrics_keys))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, m in enumerate(models):
        overall = results[m][task]["overall"]
        vals = [overall.get(k, float("nan")) for k in metrics_keys]
        bars = ax.bar(
            x + ci * width - (len(models) - 1) * width / 2,
            vals, width * 0.9,
            label=m, color=_model_color(m), edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )
    ax.set_xticks(x); ax.set_xticklabels(label_names)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title(f"Metrics comparison — {task}")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, figures_dir / f"{stem}.png")


# ── Calibration (reliability diagram) ─────────────────────────────────────────

def plot_calibration(
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    figures_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Calibración perfecta")
    for m, (y, p) in model_data.items():
        prob_true, prob_pred = get_calibration_curve(y, p, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", ms=4, color=_model_color(m), lw=1.5, label=m)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, figures_dir / f"{stem}.png")


# ── Feature importance ─────────────────────────────────────────────────────────

def plot_feature_importance(
    results: dict[str, Any],
    task: str,
    figures_dir: Path,
    stem: str,
) -> None:
    importable_models = [
        m for m in results
        if results[m].get(task, {}).get("feature_importances") is not None
    ]
    if not importable_models:
        LOGGER.info("No feature importances to plot for %s.", task)
        return

    n = len(importable_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ci, m in enumerate(importable_models):
        task_res = results[m][task]
        names = task_res["feature_names"]
        imps  = np.asarray(task_res["feature_importances"])
        order = np.argsort(imps)
        ax = axes[0, ci]
        ax.barh(
            [names[i] for i in order], imps[order],
            color=_model_color(m), edgecolor="white", linewidth=0.4,
        )
        ax.set_xlabel("Importance"); ax.set_title(m, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Feature importances — {task}", fontsize=11)
    fig.tight_layout()
    _save(fig, figures_dir / f"{stem}.png")


# ── Error maps (with external labels) ──────────────────────────────────────────

def plot_error_maps(
    results: dict[str, Any],
    key: str,
    ext_label_arr: np.ndarray,
    task: str,
    label_path_key: str,
    figures_dir: Path,
    stem: str,
) -> None:
    import rioxarray as rxr
    flood_models = [m for m in results if results[m].get(task, {}).get("per_window")]
    if not flood_models:
        return
    n = len(flood_models)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7), squeeze=False)
    fig.suptitle(f"Error maps — {task} — {key}", fontsize=11)
    for ci, m in enumerate(flood_models):
        per_w = results[m][task]["per_window"]
        path  = next((wp.get(label_path_key) for wp in per_w if wp["key"] == key), None)
        if not path or not Path(path).exists():
            for ri in range(2):
                axes[ri, ci].text(0.5, 0.5, "N/A", ha="center", va="center",
                                  transform=axes[ri, ci].transAxes)
            continue
        try:
            p_arr = rxr.open_rasterio(path, masked=True).squeeze(drop=True).values.astype(
                np.float32
            )
        except Exception:  # noqa: BLE001
            LOGGER.error("Error loading raster for error_maps: %s", path)
            continue
        p_arr = np.clip(p_arr, 0, 1)
        ext = ext_label_arr.astype(np.float32)

        ax = axes[0, ci]
        abs_err = np.abs(p_arr - ext)
        im = ax.imshow(abs_err, cmap=_ERROR_CMAP, vmin=0, vmax=1, origin="upper")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(m, fontsize=9, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel("|pred − label|", fontsize=9)

        ax = axes[1, ci]
        thr = _get_per_window_metric(results, m, task, key, "threshold")
        y_hat = (p_arr >= (thr if np.isfinite(thr) else 0.5)).astype(np.uint8)
        rgb = np.zeros((*ext.shape, 3), dtype=np.uint8)
        tp = (y_hat == 1) & (ext == 1)
        fp = (y_hat == 1) & (ext == 0)
        fn = (y_hat == 0) & (ext == 1)
        rgb[tp] = [0, 200, 0]
        rgb[fp] = [255, 80, 0]
        rgb[fn] = [30, 30, 255]
        ax.imshow(rgb, origin="upper")
        patches = [
            mpatches.Patch(color=np.array([0, 200, 0]) / 255, label="TP"),
            mpatches.Patch(color=np.array([255, 80, 0]) / 255, label="FP"),
            mpatches.Patch(color=np.array([30, 30, 255]) / 255, label="FN"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="upper right")
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel("TP/FP/FN", fontsize=9)

    fig.tight_layout()
    _save(fig, figures_dir / f"{stem}.png")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_visualize_compare(
    cfg: AppConfig,
    results: dict[str, Any],
    paths: dict[str, Path],
    targets_meta: list[dict],
) -> None:
    figures_dir = paths["run_dir"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Reference labels restricted to the TEST fold, so curves match the reported metrics
    def _labels_for(target: dict, task: str) -> np.ndarray:
        arr = _load_reference_label(target.get("labels_path"), task)
        if arr is None:
            return np.array([], dtype=np.int16)
        test = split_for_cfg(arr.shape, cfg) == SPLIT_TEST
        return np.where(test, arr, -1).astype(np.int16)

    water_label_map = {t["key"]: _labels_for(t, "water") for t in targets_meta}
    flood_label_map = {t["key"]: _labels_for(t, "flood") for t in targets_meta}

    # ── A: SAR context (antes / después / cambio) ─────────────────────────────
    for target in targets_meta:
        try:
            plot_sar_context(target, target["key"], figures_dir)
        except Exception:
            LOGGER.exception("plot_sar_context falló para key=%s", target["key"])

    # ── B: Model predictions grid ─────────────────────────────────────────────
    for target in targets_meta:
        try:
            plot_model_predictions(results, target, target["key"], figures_dir)
        except Exception:
            LOGGER.exception("plot_model_predictions falló para key=%s", target["key"])

    # ── C: Ensemble analysis ──────────────────────────────────────────────────
    for target in targets_meta:
        try:
            plot_ensemble_analysis(results, target, target["key"], figures_dir)
        except Exception:
            LOGGER.exception("plot_ensemble_analysis falló para key=%s", target["key"])

    # ── Collect concatenated labels/probs for curve plots ─────────────────────
    water_model_data = _collect_labels_probs(
        results, "water", "p_agua", water_label_map
    )
    flood_model_data = _collect_labels_probs(
        results, "flood", "p_inundacion_dado_agua", flood_label_map
    )

    # ── ROC curves ────────────────────────────────────────────────────────────
    if water_model_data:
        try:
            plot_roc_curves(water_model_data, "ROC — Water model", figures_dir, "roc_water")
        except Exception:
            LOGGER.exception("ROC water failed")
    if flood_model_data:
        try:
            plot_roc_curves(flood_model_data, "ROC — Flood model", figures_dir, "roc_flood")
        except Exception:
            LOGGER.exception("ROC flood failed")

    # ── PR curves ─────────────────────────────────────────────────────────────
    if water_model_data:
        try:
            plot_pr_curves(water_model_data, "PR — Water model", figures_dir, "pr_water")
        except Exception:
            LOGGER.exception("PR water failed")
    if flood_model_data:
        try:
            plot_pr_curves(flood_model_data, "PR — Flood model", figures_dir, "pr_flood")
        except Exception:
            LOGGER.exception("PR flood failed")

    # ── Metric bar charts ─────────────────────────────────────────────────────
    try:
        plot_metrics_bar(results, "water", figures_dir, "metrics_bar_water")
    except Exception:
        LOGGER.exception("Metrics bar water failed")
    try:
        plot_metrics_bar(results, "flood", figures_dir, "metrics_bar_flood")
    except Exception:
        LOGGER.exception("Metrics bar flood failed")

    # ── Calibration diagrams ──────────────────────────────────────────────────
    if water_model_data:
        try:
            plot_calibration(
                water_model_data, "Calibración — Agua", figures_dir, "calibration_water"
            )
        except Exception:
            LOGGER.exception("Calibration water failed")
    if flood_model_data:
        try:
            plot_calibration(
                flood_model_data, "Calibración — Inundación|Agua",
                figures_dir, "calibration_flood",
            )
        except Exception:
            LOGGER.exception("Calibration flood failed")

    # ── Feature importance ────────────────────────────────────────────────────
    try:
        plot_feature_importance(results, "water", figures_dir, "feature_importance_water")
    except Exception:
        LOGGER.exception("Feature importance water failed")
    try:
        plot_feature_importance(results, "flood", figures_dir, "feature_importance_flood")
    except Exception:
        LOGGER.exception("Feature importance flood failed")

    # ── Error maps (if external flood label exists) ───────────────────────────
    if cfg.external_labels.flood:
        ext_path = cfg.resolve_path(cfg.external_labels.flood)
        if ext_path.exists():
            try:
                ext_arr = _load_external_label(ext_path, results)
                if ext_arr is not None:
                    for target in targets_meta:
                        try:
                            plot_error_maps(
                                results,
                                target["key"],
                                ext_arr,
                                task="flood",
                                label_path_key="p_inundacion_dado_agua",
                                figures_dir=figures_dir,
                                stem=f"error_map_{target['key']}",
                            )
                        except Exception:
                            LOGGER.exception("Error map key=%s failed", target["key"])
            except Exception:
                LOGGER.exception("External label load failed")

    LOGGER.info("Todas las figuras guardadas en %s", figures_dir)
