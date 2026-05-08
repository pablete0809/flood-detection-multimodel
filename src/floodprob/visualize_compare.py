from __future__ import annotations

"""Publication-quality figures for the model comparison framework.

Figures are saved to outputs/{run_name}/figures/ as both PNG (300 dpi) and PDF.

Public entry point:
    run_visualize_compare(cfg, results, paths, targets_meta)

Individual figure functions can also be called directly.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401

from floodprob.compare_metrics import get_roc_curve, get_pr_curve, get_calibration_curve
from floodprob.config import AppConfig

LOGGER = logging.getLogger(__name__)

# ── Colour palette per model family ──────────────────────────────────────────
_MODEL_COLORS = {
    "threshold_change": "#555555",
    "logreg":           "#2196F3",
    "random_forest":    "#4CAF50",
    "xgboost":          "#FF9800",
    "mlp":              "#9C27B0",
    "unet":             "#F44336",
}
_DEFAULT_COLOR = "#888888"

_PROB_CMAP   = "RdYlBu_r"   # blue = high water, red = dry
_DIFF_CMAP   = "bwr"        # diverging, centred at 0
_ERROR_CMAP  = "hot_r"      # for |pred − label| maps


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
        res = float(abs(da.x[1] - da.x[0]))  # pixel size in map units
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


# ── A: Spatial grid ──────────────────────────────────────────────────────────

def plot_spatial_grid(
    results: dict[str, Any],
    key: str,
    cfg: AppConfig,
    figures_dir: Path,
) -> None:
    """Rows: p_agua / p_inundacion / diff_vs_best.  Columns: models."""

    water_models = [m for m in results if results[m].get("water")]
    flood_models = [m for m in results if results[m].get("flood")]
    all_models   = list(dict.fromkeys(water_models + flood_models))
    if not all_models:
        return

    # ── Load rasters for this key ──────────────────────────────────────────
    import rioxarray as rxr

    def _load(path: str | None) -> xr.DataArray | None:
        if not path or not Path(path).exists():
            return None
        try:
            return rxr.open_rasterio(path, masked=True).squeeze(drop=True)
        except Exception:  # noqa: BLE001
            return None

    water_maps: dict[str, xr.DataArray | None] = {}
    flood_maps: dict[str, xr.DataArray | None] = {}
    for m in all_models:
        # water
        pw_path = next(
            (wp["p_agua"] for wp in results[m].get("water", {}).get("per_window", []) if wp["key"] == key),
            None,
        )
        water_maps[m] = _load(pw_path)
        # flood (dado agua)
        pf_path = next(
            (wp["p_inundacion_dado_agua"] for wp in results[m].get("flood", {}).get("per_window", []) if wp["key"] == key),
            None,
        )
        flood_maps[m] = _load(pf_path)

    # ── Best water model (highest AUC-ROC for this key) ───────────────────
    best_water_auc = -1.0
    best_water_da: xr.DataArray | None = None
    for m, da in water_maps.items():
        if da is None:
            continue
        per_w = results[m].get("water", {}).get("per_window", [])
        auc = next((wp["metrics"].get("auc_roc", -1) for wp in per_w if wp["key"] == key), -1)
        if auc > best_water_auc:
            best_water_auc = auc
            best_water_da = da

    n_cols = len(all_models)
    n_rows = 3  # p_agua / p_inundacion / diff
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"Model comparison — window {key}", fontsize=12, y=1.01)

    row_labels = ["$P(agua)$", "$P(inund.|agua)$", "$\\Delta$ vs. best water"]
    for ri, rlabel in enumerate(row_labels):
        axes[ri, 0].set_ylabel(rlabel, fontsize=9)

    for ci, m in enumerate(all_models):
        axes[0, ci].set_title(m, fontsize=9, fontweight="bold")

        # Row 0: p_agua
        da_w = water_maps[m]
        ax = axes[0, ci]
        if da_w is not None:
            im = ax.imshow(da_w.values, cmap=_PROB_CMAP, vmin=0, vmax=1, origin="upper")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _add_north_arrow(ax)
            _add_scale_bar(ax, da_w)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 1: p_inundacion
        da_f = flood_maps[m]
        ax = axes[1, ci]
        if da_f is not None:
            im = ax.imshow(da_f.values, cmap=_PROB_CMAP, vmin=0, vmax=1, origin="upper")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _add_north_arrow(ax)
            _add_scale_bar(ax, da_f)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 2: difference vs best water
        ax = axes[2, ci]
        if da_w is not None and best_water_da is not None and m != "threshold_change":
            try:
                ref = best_water_da.rio.reproject_match(da_w) if da_w.shape != best_water_da.shape else best_water_da
                diff = da_w.values - ref.values
                vabs = max(float(np.nanpercentile(np.abs(diff), 98)), 0.01)
                im = ax.imshow(diff, cmap=_DIFF_CMAP, vmin=-vabs, vmax=vabs, origin="upper")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except Exception:  # noqa: BLE001
                ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    _save(fig, figures_dir / f"spatial_grid_{key}.png")


# ── B1: ROC curves ────────────────────────────────────────────────────────────

def _collect_labels_probs(
    results: dict[str, Any],
    task: str,  # "water" or "flood"
    label_key: str,  # "p_agua" or "p_inundacion_dado_agua"
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
            key = wp["key"]
            path = wp.get(label_key)
            if not path or not Path(path).exists():
                continue
            if key not in label_arr_map:
                continue
            try:
                p_arr = rxr.open_rasterio(path, masked=True).squeeze(drop=True).values.astype(np.float32)
            except Exception:  # noqa: BLE001
                continue
            y_arr = label_arr_map[key]
            mask = (y_arr >= 0) & np.isfinite(p_arr)
            if mask.sum() == 0:
                continue
            ys.append(y_arr[mask].astype(np.uint8))
            ps.append(np.clip(p_arr[mask], 0, 1))
        if ys:
            out[m] = (np.concatenate(ys), np.concatenate(ps))
    return out


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


# ── B2: Metric bar chart ──────────────────────────────────────────────────────

def plot_metrics_bar(
    results: dict[str, Any],
    task: str,
    figures_dir: Path,
    stem: str,
) -> None:
    metrics_keys = ["auc_roc", "f1", "iou"]
    label_names  = ["AUC-ROC", "F1@0.5", "IoU"]
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
        # Value labels
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


# ── B3: Calibration (reliability diagram) ─────────────────────────────────────

def plot_calibration(
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    figures_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    for m, (y, p) in model_data.items():
        prob_true, prob_pred = get_calibration_curve(y, p, n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", ms=4, color=_model_color(m), lw=1.5, label=m)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, figures_dir / f"{stem}.png")


# ── D: Feature importance ─────────────────────────────────────────────────────

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


# ── C: Error maps (with external labels) ──────────────────────────────────────

def plot_error_maps(
    results: dict[str, Any],
    key: str,
    ext_label_arr: np.ndarray,  # (H, W) binary
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
                axes[ri, ci].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[ri, ci].transAxes)
            continue
        try:
            p_arr = rxr.open_rasterio(path, masked=True).squeeze(drop=True).values.astype(np.float32)
        except Exception:  # noqa: BLE001
            continue
        p_arr = np.clip(p_arr, 0, 1)
        ext = ext_label_arr.astype(np.float32)

        # Absolute error
        ax = axes[0, ci]
        abs_err = np.abs(p_arr - ext)
        im = ax.imshow(abs_err, cmap=_ERROR_CMAP, vmin=0, vmax=1, origin="upper")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(m, fontsize=9, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel("|pred − label|", fontsize=9)

        # TP/FP/FN colour map (at threshold 0.5)
        ax = axes[1, ci]
        y_hat = (p_arr >= 0.5).astype(np.uint8)
        rgb = np.zeros((*ext.shape, 3), dtype=np.uint8)
        tp = (y_hat == 1) & (ext == 1)
        fp = (y_hat == 1) & (ext == 0)
        fn = (y_hat == 0) & (ext == 1)
        rgb[tp] = [0, 200, 0]   # green = TP
        rgb[fp] = [255, 80, 0]  # orange = FP
        rgb[fn] = [30, 30, 255] # blue = FN
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

    # Collect per-key label arrays for curve plots
    import xarray as xr
    def _labels_for(target: dict, var: str) -> np.ndarray:
        try:
            ds = xr.open_dataset(Path(target["labels_path"]))
            return ds[var].values.astype(np.int16)
        except Exception:  # noqa: BLE001
            return np.array([], dtype=np.int16)

    water_label_map  = {t["key"]: _labels_for(t, "water_label")  for t in targets_meta}
    flood_label_map  = {t["key"]: _labels_for(t, "flood_label")  for t in targets_meta}

    # ── A: Spatial grids per window ──────────────────────────────────────────
    for target in targets_meta:
        try:
            plot_spatial_grid(results, target["key"], cfg, figures_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Spatial grid key=%s failed: %s", target["key"], exc)

    # ── Collect concatenated labels/probs for curve plots ──────────────────
    water_model_data = _collect_labels_probs(results, "water", "p_agua",                  water_label_map)
    flood_model_data = _collect_labels_probs(results, "flood", "p_inundacion_dado_agua",  flood_label_map)

    # ── B1: ROC curves ────────────────────────────────────────────────────────
    if water_model_data:
        try:
            plot_roc_curves(water_model_data, "ROC — Water model", figures_dir, "roc_water")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ROC water failed: %s", exc)
    if flood_model_data:
        try:
            plot_roc_curves(flood_model_data, "ROC — Flood model", figures_dir, "roc_flood")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ROC flood failed: %s", exc)

    # ── B1b: PR curves ────────────────────────────────────────────────────────
    if water_model_data:
        try:
            plot_pr_curves(water_model_data, "PR — Water model", figures_dir, "pr_water")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("PR water failed: %s", exc)
    if flood_model_data:
        try:
            plot_pr_curves(flood_model_data, "PR — Flood model", figures_dir, "pr_flood")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("PR flood failed: %s", exc)

    # ── B2: Metric bar charts ─────────────────────────────────────────────────
    try:
        plot_metrics_bar(results, "water", figures_dir, "metrics_bar_water")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Metrics bar water failed: %s", exc)
    try:
        plot_metrics_bar(results, "flood", figures_dir, "metrics_bar_flood")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Metrics bar flood failed: %s", exc)

    # ── B3: Calibration diagrams ──────────────────────────────────────────────
    if water_model_data:
        try:
            plot_calibration(water_model_data, "Calibration — Water", figures_dir, "calibration_water")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Calibration water failed: %s", exc)
    if flood_model_data:
        try:
            plot_calibration(flood_model_data, "Calibration — Flood|Water", figures_dir, "calibration_flood")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Calibration flood failed: %s", exc)

    # ── D: Feature importance ─────────────────────────────────────────────────
    try:
        plot_feature_importance(results, "water", figures_dir, "feature_importance_water")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature importance water failed: %s", exc)
    try:
        plot_feature_importance(results, "flood", figures_dir, "feature_importance_flood")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Feature importance flood failed: %s", exc)

    # ── C: Error maps (if external flood label exists) ────────────────────────
    if cfg.external_labels.flood:
        ext_path = cfg.resolve_path(cfg.external_labels.flood)
        if ext_path.exists():
            try:
                import rioxarray as rxr
                ext_da = rxr.open_rasterio(str(ext_path), masked=True).squeeze(drop=True)
                ext_arr = (ext_da.values > 0).astype(np.uint8)
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
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("Error map key=%s failed: %s", target["key"], exc)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("External label load failed: %s", exc)

    LOGGER.info("All figures saved to %s", figures_dir)
