from __future__ import annotations

"""Multi-model comparison orchestrator.

Usage:
    from floodprob.compare import run_compare
    report = run_compare(cfg)

The function:
  1. Loads the preprocess manifest.
  2. Splits every window into spatial blocks (floodprob.splits): train / val / test.
  3. For each model in cfg.model.compare.models_water / models_flood:
       a. Trains a bundle on TRAIN pixels only, with labels chosen by
          cfg.model.train_on ("pseudo" or "gt"), or loads a cached .joblib.
       b. Runs inference → (p_agua, p_std) / (p_inundacion, p_std) DataArrays.
       c. Saves GeoTIFFs: p_agua_{model}_{key}.tif etc.
       d. Picks the F1-optimal threshold on VAL pixels (training-type labels) and
          computes metrics on TEST pixels against the reference labels (ground truth
          when available). Every model is scored on the same pixel set.
  4. Aggregates into manifest_compare.json.
  5. Calls visualize_compare.run_visualize_compare(cfg, results).

Outputs go to outputs/<run_name>/compare_<train_on>/ so that weakly- and fully-
supervised runs do not overwrite each other.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from floodprob.compare_metrics import aggregate_metrics, best_f1_threshold, compute_all_metrics
from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import reference_labels, training_labels
from floodprob.models import train_flood_given_water, train_water
from floodprob.models import train_rf, train_mlp, train_xgboost_water
from floodprob.splits import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, mask_to_train, split_for_cfg
from floodprob.utils.io import build_run_dirs, ensure_dir, read_json, write_json
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

    def _train_attention_unet_flood(water_sets, flood_sets):
        from floodprob.models import train_attention_unet
        return None, train_attention_unet.train_flood_attention_unet_bundle(
            flood_sets, cfg, bundle_dir=bundle_dir
        )

    def _train_deeplabv3plus_flood(water_sets, flood_sets):
        from floodprob.models import train_deeplabv3plus
        return None, train_deeplabv3plus.train_flood_deeplabv3plus_bundle(
            flood_sets, cfg, bundle_dir=bundle_dir
        )

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
        "attention_unet": {
            "tasks": ["flood"],
            "train_flood": _train_attention_unet_flood,
            "predict_flood": _predict_attention_unet_flood,
        },
        "deeplabv3plus": {
            "tasks": ["flood"],
            "train_flood": _train_deeplabv3plus_flood,
            "predict_flood": _predict_deeplabv3plus_flood,
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


def _predict_attention_unet_flood(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
    p_water: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    from floodprob.models import train_attention_unet
    return train_attention_unet.predict_flood_attention_unet(feature_ds, bundle)


def _predict_deeplabv3plus_flood(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
    p_water: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    from floodprob.models import train_deeplabv3plus
    return train_deeplabv3plus.predict_flood_deeplabv3plus(feature_ds, bundle)


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

def _feature_valid_mask(feat_ds: xr.Dataset) -> np.ndarray:
    """Pixels where every 2-D feature is finite: the common evaluation support."""
    arrays = [feat_ds[v].values for v in feat_ds.data_vars if feat_ds[v].ndim == 2]
    return np.logical_and.reduce([np.isfinite(a) for a in arrays])


def _eval_window(
    y_ref: np.ndarray,     # (H, W) reference labels (ground truth if available)
    y_sel: np.ndarray,     # (H, W) labels of the training type, used to pick the threshold
    p_prob: np.ndarray,    # (H, W) float probabilities
    split: np.ndarray,     # (H, W) SPLIT_* codes
    valid: np.ndarray,     # (H, W) common feature-valid mask
) -> dict[str, float]:
    """Threshold chosen on VAL pixels, metrics computed on TEST pixels."""
    finite = np.isfinite(p_prob)
    val = (split == SPLIT_VAL) & (y_sel >= 0) & valid & finite
    threshold = best_f1_threshold(y_sel[val].astype(np.uint8), p_prob[val])

    test = (split == SPLIT_TEST) & (y_ref >= 0) & valid
    n_missing = int((test & ~finite).sum())
    if n_missing:
        LOGGER.warning("%d test pixels have no prediction and are excluded", n_missing)
    test &= finite
    if test.sum() == 0:
        return {}

    y = y_ref[test].astype(np.uint8)
    p = np.clip(p_prob[test], 0, 1)
    metrics = compute_all_metrics(y, p, threshold=threshold)
    at_05 = compute_all_metrics(y, p, threshold=0.5)
    metrics.update(
        threshold=threshold,
        f1_at_05=at_05["f1"],
        iou_at_05=at_05["iou"],
        n_test=float(test.sum()),
        n_test_pos=float(y.sum()),
        n_missing_pred=float(n_missing),
    )
    return metrics


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_compare(cfg: AppConfig, force_retrain: bool = False) -> dict[str, Any]:
    run_paths = build_run_dirs(cfg.outputs.out_dir, cfg.run_name)
    train_on = cfg.model.train_on

    # Load preprocess manifest
    manifest_path = run_paths["metadata"] / "manifest_preprocess.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Preprocess manifest not found: {manifest_path}. Run `floodprob preprocess` first."
        )
    pre = read_json(manifest_path)

    exp_dir = ensure_dir(run_paths["run_dir"] / f"compare_{train_on}")
    paths = {
        "run_dir": exp_dir,
        **{name: ensure_dir(exp_dir / name) for name in ("metadata", "models", "predictions", "figures")},
    }
    LOGGER.info("Compare experiment: train_on=%s → %s", train_on, exp_dir)

    # Build train sets (TRAIN pixels only) and per-window evaluation context
    water_train_sets: list[tuple[xr.Dataset, xr.DataArray]] = []
    flood_train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]] = []
    targets_meta: list[dict] = []
    eval_ctx: dict[str, dict[str, Any]] = {}

    for target in pre["targets"]:
        feat_ds = xr.open_dataset(Path(target["features_path"]))
        labels_ds = xr.open_dataset(Path(target["labels_path"]))
        split = split_for_cfg(labels_ds["flood_label"].shape, cfg)

        water_y = mask_to_train(training_labels(labels_ds, "water", train_on), split)
        flood_y = mask_to_train(training_labels(labels_ds, "flood", train_on), split)
        water_train_sets.append((feat_ds, water_y))
        flood_train_sets.append((feat_ds, flood_y, labels_ds["p_water_initial"]))
        targets_meta.append(target)

        eval_ctx[target["key"]] = {"split": split, "valid": _feature_valid_mask(feat_ds)}
        LOGGER.info(
            "Window %s: split px train=%d val=%d test=%d | flood train labels pos=%d neg=%d",
            target["key"],
            int((split == SPLIT_TRAIN).sum()), int((split == SPLIT_VAL).sum()),
            int((split == SPLIT_TEST).sum()),
            int((flood_y.values == 1).sum()), int((flood_y.values == 0).sum()),
        )

    def _labels_for_eval(labels_ds: xr.Dataset, task: str) -> tuple[np.ndarray, np.ndarray, str]:
        y_ref, source = reference_labels(labels_ds, task)
        y_sel = training_labels(labels_ds, task, train_on)
        return y_ref.values.astype(np.int16), y_sel.values.astype(np.int16), source

    registry = _build_registry(cfg, paths["models"])
    compare_cfg = cfg.model.compare

    # Collect results per model (only models that appear in at least one task list)
    active_models = set(compare_cfg.models_water) | set(compare_cfg.models_flood)
    results: dict[str, Any] = {
        m: {"water": {}, "flood": {}} for m in registry if m in active_models
    }

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
            y_ref, y_sel, label_source = _labels_for_eval(labels_ds, "water")
            metrics = _eval_window(y_ref, y_sel, p_agua.values, **eval_ctx[key])
            per_window_metrics.append(metrics)
            window_predictions.append({
                "key": key,
                "p_agua": str(tif_path),
                "p_agua_std": str(std_path) if std_path else None,
                "label_source": label_source,
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

            y_ref, y_sel, label_source = _labels_for_eval(labels_ds, "flood")
            metrics = _eval_window(y_ref, y_sel, p_flood_gw.values, **eval_ctx[key])
            per_window_metrics.append(metrics)
            window_predictions.append({
                "key": key,
                "p_inundacion_dado_agua": str(tif_path),
                "p_inundacion_dado_agua_std": str(std_path) if std_path else None,
                "label_source": label_source,
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
        "protocol": {
            "train_on": train_on,
            "split": {
                "strategy": "random_spatial_blocks",
                "tile_size_px": cfg.model.tile_size,
                "n_folds": cfg.model.n_folds,
                "test_fold": cfg.model.test_fold,
                "val_fold": cfg.model.val_fold,
                "seed": cfg.seed,
            },
            "threshold": "F1-optimal on VAL fold with training-type labels",
            "evaluation": "TEST fold, reference labels (gt if available), common feature-valid mask",
        },
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
