from __future__ import annotations

"""Multi-model comparison orchestrator.

Usage:
    from floodprob.compare import run_compare
    report = run_compare(cfg)

The function:
  1. Loads the preprocess manifest.
  2. For each model in cfg.model.compare.models_water / models_flood:
       a. Trains a bundle (or loads cached .joblib if already exists).
       b. Runs inference → (p_agua, p_std) / (p_inundacion, p_std) DataArrays.
       c. Saves GeoTIFFs: p_agua_{model}_{key}.tif etc.
       d. Computes metrics vs pseudo-labels (pixels with label != -1).
  3. Aggregates into manifest_compare.json.
  4. Calls visualize_compare.run_visualize_compare(cfg, results).
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from floodprob.compare_metrics import compute_all_metrics, aggregate_metrics
from floodprob.config import AppConfig
from floodprob.models import train_flood_given_water, train_water
from floodprob.models import train_rf, train_mlp, train_xgboost_water
from floodprob.utils.io import build_run_dirs, read_json, write_json
from floodprob.utils.geospatial import save_geotiff

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_registry(cfg: AppConfig, bundle_dir: Path) -> dict[str, dict[str, Any]]:
    """Return a dict keyed by model name with train/predict functions."""

    # --- Water models ---
    def _train_threshold(water_sets, flood_sets):
        tmp_cfg = cfg.model_copy(deep=True)
        tmp_cfg.model.water_model = "threshold_change"
        return train_water.train_water_model_bundle(water_sets, tmp_cfg), None

    def _train_logreg_water(water_sets, flood_sets):
        tmp_cfg = cfg.model_copy(deep=True)
        tmp_cfg.model.water_model = "logreg"
        return train_water.train_water_model_bundle(water_sets, tmp_cfg), None

    def _train_rf_water(water_sets, flood_sets):
        return train_rf.train_water_rf_bundle(water_sets, cfg), None

    def _train_xgb_water(water_sets, flood_sets):
        return train_xgboost_water.train_water_xgboost_bundle(water_sets, cfg), None

    def _train_mlp_water(water_sets, flood_sets):
        return train_mlp.train_water_mlp_bundle(water_sets, cfg), None

    # --- Flood models ---
    def _train_logreg_flood(water_sets, flood_sets):
        tmp_cfg = cfg.model_copy(deep=True)
        tmp_cfg.model.flood_given_water_model = "logreg"
        return None, train_flood_given_water.train_flood_model_bundle(flood_sets, tmp_cfg)

    def _train_rf_flood(water_sets, flood_sets):
        return None, train_rf.train_flood_rf_bundle(flood_sets, cfg)

    def _train_xgb_flood(water_sets, flood_sets):
        return None, train_xgboost_water.train_flood_xgboost_bundle(flood_sets, cfg)

    def _train_mlp_flood(water_sets, flood_sets):
        return None, train_mlp.train_flood_mlp_bundle(flood_sets, cfg)

    def _train_unet_flood(water_sets, flood_sets):
        from floodprob.models import train_unet
        return None, train_unet.train_flood_unet_bundle(flood_sets, cfg, bundle_dir=bundle_dir)

    return {
        # water-only models
        "threshold_change": {
            "tasks": ["water"],
            "train": _train_threshold,
            "predict_water": _predict_generic_water,
        },
        "logreg": {
            "tasks": ["water", "flood"],
            "train_water": _train_logreg_water,
            "train_flood": _train_logreg_flood,
            "predict_water": _predict_generic_water,
            "predict_flood": _predict_generic_flood,
        },
        "random_forest": {
            "tasks": ["water", "flood"],
            "train_water": _train_rf_water,
            "train_flood": _train_rf_flood,
            "predict_water": _predict_generic_water,
            "predict_flood": _predict_generic_flood,
        },
        "xgboost": {
            "tasks": ["water", "flood"],
            "train_water": _train_xgb_water,
            "train_flood": _train_xgb_flood,
            "predict_water": _predict_generic_water,
            "predict_flood": _predict_generic_flood,
        },
        "mlp": {
            "tasks": ["water", "flood"],
            "train_water": _train_mlp_water,
            "train_flood": _train_mlp_flood,
            "predict_water": _predict_generic_water,
            "predict_flood": _predict_generic_flood,
        },
        "unet": {
            "tasks": ["flood"],
            "train_flood": _train_unet_flood,
            "predict_flood": _predict_unet_flood,
        },
    }


# ---------------------------------------------------------------------------
# Generic predict helpers
# ---------------------------------------------------------------------------

def _predict_generic_water(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray | None]:
    return train_water.predict_water_probability(feature_ds, bundle)


def _predict_generic_flood(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
    p_water: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    _p_water = p_water if p_water is not None else xr.ones_like(feature_ds[bundle["feature_names"][0]])
    return train_flood_given_water.predict_flood_probability(feature_ds, _p_water, bundle)


def _predict_unet_flood(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
    p_water: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    from floodprob.models import train_unet
    return train_unet.predict_flood_unet(feature_ds, bundle)


# ---------------------------------------------------------------------------
# Bundle cache helpers
# ---------------------------------------------------------------------------

def _bundle_path(paths: dict[str, Path], model_name: str, task: str) -> Path:
    return paths["models"] / f"compare_{model_name}_{task}.joblib"


def _load_or_train(
    model_name: str,
    task: str,
    train_fn,
    water_sets,
    flood_sets,
    paths: dict[str, Path],
    force_retrain: bool = False,
) -> dict[str, Any]:
    cache_path = _bundle_path(paths, model_name, task)
    if not force_retrain and cache_path.exists():
        LOGGER.info("Loading cached bundle: %s", cache_path)
        return joblib.load(cache_path)
    LOGGER.info("Training %s (%s)…", model_name, task)
    water_bundle, flood_bundle = train_fn(water_sets, flood_sets)
    bundle = water_bundle if task == "water" else flood_bundle
    if bundle is not None:
        joblib.dump(bundle, cache_path)
    return bundle


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _eval_window(
    y_labels: np.ndarray,  # (H, W) int labels
    p_prob: np.ndarray,    # (H, W) float probabilities
    tile_size: int,
) -> dict[str, float]:
    """Evaluate one model×window using spatial holdout (tile modulo 5)."""
    H, W = y_labels.shape
    y_idx = np.arange(H)[:, None]
    x_idx = np.arange(W)[None, :]
    holdout = ((y_idx // tile_size) * 10000 + (x_idx // tile_size)) % 5 == 0

    mask = (y_labels >= 0) & np.isfinite(p_prob) & holdout
    if mask.sum() == 0:
        return {}
    return compute_all_metrics(y_labels[mask].astype(np.uint8), np.clip(p_prob[mask], 0, 1))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_compare(cfg: AppConfig, force_retrain: bool = False) -> dict[str, Any]:
    paths = build_run_dirs(cfg.outputs.out_dir, cfg.run_name)
    # Add figures sub-dir
    figures_dir = paths["run_dir"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocess manifest
    manifest_path = paths["metadata"] / "manifest_preprocess.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Preprocess manifest not found: {manifest_path}. Run `floodprob preprocess` first."
        )
    pre = read_json(manifest_path)

    # Build train sets
    water_train_sets: list[tuple[xr.Dataset, xr.DataArray]] = []
    flood_train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]] = []
    targets_meta: list[dict] = []

    for target in pre["targets"]:
        feat_ds = xr.open_dataset(Path(target["features_path"]))
        labels_ds = xr.open_dataset(Path(target["labels_path"]))
        water_train_sets.append((feat_ds, labels_ds["water_label"]))
        flood_train_sets.append((feat_ds, labels_ds["flood_label"], labels_ds["p_water_initial"]))
        targets_meta.append(target)

    registry = _build_registry(cfg, paths["models"])
    compare_cfg = cfg.model.compare

    # Collect results per model
    results: dict[str, Any] = {m: {"water": {}, "flood": {}} for m in registry}

    # ---- Train + predict water models ----
    for model_name in compare_cfg.models_water:
        if model_name not in registry:
            LOGGER.warning("Unknown model '%s' in models_water — skipping.", model_name)
            continue
        info = registry[model_name]
        train_fn = info.get("train_water") or info.get("train")
        if train_fn is None:
            LOGGER.warning("Model '%s' has no water train function — skipping.", model_name)
            continue

        bundle = _load_or_train(
            model_name, "water", train_fn, water_train_sets, flood_train_sets, paths, force_retrain
        )
        if bundle is None:
            continue

        predict_fn = info["predict_water"]
        per_window_metrics: list[dict] = []
        window_predictions: list[dict] = []

        for target in targets_meta:
            key = target["key"]
            feat_ds = xr.open_dataset(Path(target["features_path"]))
            labels_ds = xr.open_dataset(Path(target["labels_path"]))

            try:
                p_agua, p_agua_std = predict_fn(feat_ds, bundle)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Predict water %s key=%s failed: %s", model_name, key, exc)
                continue

            # Save GeoTIFF
            tif_path = paths["predictions"] / f"p_agua_{model_name}_{key}.tif"
            save_geotiff(p_agua, str(tif_path), fallback_crs=cfg.crs_salida)
            std_path = None
            if p_agua_std is not None:
                std_path = paths["predictions"] / f"p_agua_{model_name}_{key}_std.tif"
                save_geotiff(p_agua_std, str(std_path), fallback_crs=cfg.crs_salida)

            # Metrics
            water_labels = labels_ds["water_label"].values.astype(np.int16)
            metrics = _eval_window(water_labels, p_agua.values, cfg.model.tile_size)
            per_window_metrics.append(metrics)
            window_predictions.append({
                "key": key,
                "p_agua": str(tif_path),
                "p_agua_std": str(std_path) if std_path else None,
                "metrics": metrics,
            })
            LOGGER.info("Water %s key=%s  → %s", model_name, key, metrics)

        results[model_name]["water"] = {
            "bundle_path": str(_bundle_path(paths, model_name, "water")),
            "feature_names": bundle["feature_names"],
            "feature_importances": bundle.get("feature_importances"),
            "per_window": window_predictions,
            "overall": aggregate_metrics(per_window_metrics),
        }

    # ---- Determine "best" water model (highest AUC-ROC overall) for flood head masking ----
    best_water_probs: dict[str, dict[str, xr.DataArray]] = {}  # key -> {model: da}
    for model_name, model_res in results.items():
        if not model_res["water"]:
            continue
        for wp in model_res["water"].get("per_window", []):
            key = wp["key"]
            if not wp.get("p_agua"):
                continue
            import rioxarray as rxr
            da = rxr.open_rasterio(wp["p_agua"], masked=True).squeeze(drop=True)
            best_water_probs.setdefault(key, {})[model_name] = da

    # Use logreg water (or first available) as the p_water mask for flood models
    def _get_p_water_for_flood(key: str) -> xr.DataArray | None:
        pool = best_water_probs.get(key, {})
        for preferred in ["logreg", "random_forest", "threshold_change"]:
            if preferred in pool:
                return pool[preferred]
        return next(iter(pool.values())) if pool else None

    # ---- Train + predict flood models ----
    for model_name in compare_cfg.models_flood:
        if model_name not in registry:
            LOGGER.warning("Unknown model '%s' in models_flood — skipping.", model_name)
            continue
        info = registry[model_name]
        train_fn = info.get("train_flood")
        if train_fn is None:
            LOGGER.warning("Model '%s' has no flood train function — skipping.", model_name)
            continue

        bundle = _load_or_train(
            model_name, "flood", train_fn, water_train_sets, flood_train_sets, paths, force_retrain
        )
        if bundle is None:
            continue

        predict_fn = info["predict_flood"]
        per_window_metrics: list[dict] = []
        window_predictions: list[dict] = []

        for target in targets_meta:
            key = target["key"]
            feat_ds = xr.open_dataset(Path(target["features_path"]))
            labels_ds = xr.open_dataset(Path(target["labels_path"]))
            p_water = _get_p_water_for_flood(key)

            try:
                p_flood_gw, p_flood_std = predict_fn(feat_ds, bundle, p_water=p_water)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Predict flood %s key=%s failed: %s", model_name, key, exc)
                continue

            tif_path = paths["predictions"] / f"p_inundacion_{model_name}_{key}.tif"
            save_geotiff(p_flood_gw, str(tif_path), fallback_crs=cfg.crs_salida)
            std_path = None
            if p_flood_std is not None:
                std_path = paths["predictions"] / f"p_inundacion_{model_name}_{key}_std.tif"
                save_geotiff(p_flood_std, str(std_path), fallback_crs=cfg.crs_salida)

            # Final p_flood = p_agua × p_flood|agua
            p_flood_map = None
            if p_water is not None:
                try:
                    if p_flood_gw.rio.crs is None:
                        p_flood_gw = p_flood_gw.rio.write_crs(cfg.crs_salida)
                    p_w_aligned = p_water.rio.reproject_match(p_flood_gw)
                    p_flood_arr = (p_w_aligned.values * p_flood_gw.values).clip(0, 1)
                    p_flood_map = xr.DataArray(
                        p_flood_arr, coords=p_flood_gw.coords, dims=p_flood_gw.dims,
                        name="p_inundacion"
                    ).astype("float32")
                    if p_flood_gw.rio.crs:
                        p_flood_map = p_flood_map.rio.write_crs(p_flood_gw.rio.crs)
                    pfinal_path = paths["predictions"] / f"p_inundacion_final_{model_name}_{key}.tif"
                    save_geotiff(p_flood_map, str(pfinal_path), fallback_crs=cfg.crs_salida)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Could not compute final p_flood for %s key=%s: %s", model_name, key, exc)

            flood_labels = labels_ds["flood_label"].values.astype(np.int16)
            metrics = _eval_window(flood_labels, p_flood_gw.values, cfg.model.tile_size)
            per_window_metrics.append(metrics)
            window_predictions.append({
                "key": key,
                "p_inundacion_dado_agua": str(tif_path),
                "p_inundacion_dado_agua_std": str(std_path) if std_path else None,
                "metrics": metrics,
            })
            LOGGER.info("Flood %s key=%s  → %s", model_name, key, metrics)

        results[model_name]["flood"] = {
            "bundle_path": str(_bundle_path(paths, model_name, "flood")),
            "feature_names": bundle["feature_names"],
            "feature_importances": bundle.get("feature_importances"),
            "per_window": window_predictions,
            "overall": aggregate_metrics(per_window_metrics),
        }

    # ---- Write manifest ----
    report = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": cfg.run_name,
        "models": list(registry.keys()),
        "models_water_trained": compare_cfg.models_water,
        "models_flood_trained": compare_cfg.models_flood,
        "results": results,
    }
    manifest_out = paths["metadata"] / "manifest_compare.json"
    write_json(manifest_out, report)
    LOGGER.info("Compare manifest written to %s", manifest_out)

    # ---- Visualize ----
    try:
        from floodprob import visualize_compare
        visualize_compare.run_visualize_compare(cfg, results, paths, targets_meta)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Visualization failed (non-fatal): %s", exc)

    return report
