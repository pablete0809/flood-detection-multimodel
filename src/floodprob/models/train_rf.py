from __future__ import annotations

import logging
from typing import Any

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import flatten_samples
from floodprob.models.train_water import (
    WATER_FEATURE_CANDIDATES,
    _dataset_to_matrix,
)
from floodprob.models.train_flood_given_water import FLOOD_FEATURE_CANDIDATES

LOGGER = logging.getLogger(__name__)


def _rf_feature_importances(calibrated_model: CalibratedClassifierCV) -> np.ndarray:
    """Extract feature importances from a CalibratedClassifierCV wrapping an RF.

    When cv is an integer (cross-val), `calibrated_model.estimator` is the
    unfitted template; the actually-fitted RF estimators live inside
    `calibrated_model.calibrated_classifiers_[*].estimator`.
    """
    importances_list = [
        cc.estimator.feature_importances_
        for cc in calibrated_model.calibrated_classifiers_
    ]
    return np.mean(importances_list, axis=0)


def _fit_rf(
    X: np.ndarray,
    y: np.ndarray,
    calibration: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    seed: int,
) -> CalibratedClassifierCV:
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("RF model requires at least two classes")
    min_count = int(counts.min())
    cv = 3 if min_count >= 3 else 2
    method = calibration
    if method == "isotonic" and min_count < 30:
        method = "sigmoid"
        LOGGER.warning("Few samples for isotonic calibration; falling back to sigmoid.")
    model = CalibratedClassifierCV(base, method=method, cv=cv)
    model.fit(X, y)
    return model


def _collect_xy(
    train_sets_water: list[tuple[xr.Dataset, xr.DataArray]],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    X_all, y_all = [], []
    for ds, labels in train_sets_water:
        X, y = flatten_samples(ds, labels, feature_names)
        if len(y) == 0:
            continue
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        raise ValueError("No training samples after filtering")
    return np.vstack(X_all), np.concatenate(y_all)


def train_water_rf_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    feature_names = [
        f for f in WATER_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _ in train_sets)
    ]
    if "delta_vv" not in feature_names or "delta_vh" not in feature_names:
        raise ValueError("RF water model requires delta_vv and delta_vh features")
    X_train, y_train = _collect_xy(train_sets, feature_names)
    LOGGER.info("RF Water training samples: %s", len(y_train))

    rf_cfg = cfg.model.compare.random_forest
    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        model = _fit_rf(
            X_train,
            y_train,
            calibration=cfg.model.calibration,
            n_estimators=rf_cfg.n_estimators,
            max_depth=rf_cfg.max_depth,
            min_samples_leaf=rf_cfg.min_samples_leaf,
            seed=cfg.seed + idx,
        )
        ensemble_models.append(model)

    # collect feature importances from the underlying RF estimators
    importances = np.mean(
        [_rf_feature_importances(m) for m in ensemble_models], axis=0
    )
    return {
        "model_type": "random_forest",
        "feature_names": feature_names,
        "models": ensemble_models,
        "feature_importances": importances,
    }


def train_flood_rf_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    feature_names = [
        f for f in FLOOD_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _, _ in train_sets)
    ]
    required = {"delta_vv", "delta_vh"}
    if not required.issubset(set(feature_names)):
        raise ValueError(f"RF flood model requires features {sorted(required)}")

    X_all, y_all = [], []
    for ds, flood_labels, _p_water in train_sets:
        X, y = flatten_samples(ds, flood_labels, feature_names=feature_names)
        if len(y) == 0:
            continue
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        raise ValueError("No flood training samples after filtering")
    X_train, y_train = np.vstack(X_all), np.concatenate(y_all)
    LOGGER.info("RF Flood training samples: %s", len(y_train))

    rf_cfg = cfg.model.compare.random_forest
    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        model = _fit_rf(
            X_train,
            y_train,
            calibration=cfg.model.calibration,
            n_estimators=rf_cfg.n_estimators,
            max_depth=rf_cfg.max_depth,
            min_samples_leaf=rf_cfg.min_samples_leaf,
            seed=cfg.seed + idx,
        )
        ensemble_models.append(model)

    importances = np.mean(
        [_rf_feature_importances(m) for m in ensemble_models], axis=0
    )
    return {
        "model_type": "random_forest",
        "feature_names": feature_names,
        "models": ensemble_models,
        "feature_importances": importances,
    }


def predict_probability(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
    prob_name: str = "p_rf",
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Generic predict for RF water or flood bundles."""
    feature_names = bundle["feature_names"]
    missing = [n for n in feature_names if n not in feature_ds]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X, template = _dataset_to_matrix(feature_ds, feature_names)
    valid = np.isfinite(X).all(axis=1)

    probs = np.full((len(bundle["models"]), X.shape[0]), np.nan, dtype=np.float32)
    for i, model in enumerate(bundle["models"]):
        p = model.predict_proba(X[valid])[:, 1]
        probs[i, valid] = p.astype(np.float32)

    p_mean = np.nanmean(probs, axis=0)
    p_std = np.nanstd(probs, axis=0) if probs.shape[0] > 1 else None

    crs = template.rio.crs
    mean_da = xr.DataArray(
        p_mean.reshape(template.shape), coords=template.coords, dims=template.dims, name=prob_name
    ).astype("float32")
    if crs:
        mean_da = mean_da.rio.write_crs(crs)
    std_da = None
    if p_std is not None:
        std_da = xr.DataArray(
            p_std.reshape(template.shape), coords=template.coords, dims=template.dims,
            name=f"{prob_name}_std"
        ).astype("float32")
        if crs:
            std_da = std_da.rio.write_crs(crs)
    return mean_da, std_da
