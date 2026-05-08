from __future__ import annotations

"""XGBoost model for the *water* head.

The flood head already has xgboost_if_available inside train_flood_given_water.py.
This module mirrors that logic for the water head so that compare.py can use
a consistent model_type="xgboost" for both tasks.
"""

import logging
from typing import Any

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from sklearn.calibration import CalibratedClassifierCV

from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import flatten_samples
from floodprob.models.train_water import (
    WATER_FEATURE_CANDIDATES,
    _dataset_to_matrix,
    _fit_logreg,
)
from floodprob.models.train_flood_given_water import FLOOD_FEATURE_CANDIDATES

LOGGER = logging.getLogger(__name__)


def _fit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    calibration: str,
    seed: int,
) -> CalibratedClassifierCV:
    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError:
        LOGGER.warning("xgboost not installed; falling back to logistic regression for water head.")
        return _fit_logreg(X, y, calibration=calibration, c_value=1.0, max_iter=500, seed=seed)

    base = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=seed,
        n_jobs=4,
        eval_metric="logloss",
    )
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("XGBoost water model requires at least two classes")
    min_count = int(counts.min())
    cv = 3 if min_count >= 3 else 2
    method = calibration
    if method == "isotonic" and min_count < 30:
        method = "sigmoid"
    model = CalibratedClassifierCV(base, method=method, cv=cv)
    model.fit(X, y)
    return model


def train_water_xgboost_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    feature_names = [
        f for f in WATER_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _ in train_sets)
    ]
    if "delta_vv" not in feature_names or "delta_vh" not in feature_names:
        raise ValueError("XGBoost water model requires delta_vv and delta_vh features")

    X_all, y_all = [], []
    for ds, labels in train_sets:
        X, y = flatten_samples(ds, labels, feature_names)
        if len(y) == 0:
            continue
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        raise ValueError("No water training samples after filtering")
    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)
    LOGGER.info("XGBoost Water training samples: %s", len(y_train))

    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        model = _fit_xgboost(X_train, y_train, calibration=cfg.model.calibration, seed=cfg.seed + idx)
        ensemble_models.append(model)

    # Extract feature importances if xgboost was actually used
    importances = None
    try:
        importances = np.mean(
            [m.estimator.feature_importances_ for m in ensemble_models], axis=0
        )
    except AttributeError:
        pass  # fallback logreg has coef_ not feature_importances_

    return {
        "model_type": "xgboost",
        "feature_names": feature_names,
        "models": ensemble_models,
        "feature_importances": importances,
    }


def train_flood_xgboost_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    """XGBoost for the flood head — wraps the existing logic."""
    feature_names = [
        f for f in FLOOD_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _, _ in train_sets)
    ]
    required = {"delta_vv", "delta_vh"}
    if not required.issubset(set(feature_names)):
        raise ValueError(f"XGBoost flood model requires features {sorted(required)}")

    X_all, y_all = [], []
    for ds, flood_labels, _p_water in train_sets:
        X, y = flatten_samples(ds, flood_labels, feature_names=feature_names)
        if len(y) == 0:
            continue
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        raise ValueError("No flood training samples after filtering")
    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)
    LOGGER.info("XGBoost Flood training samples: %s", len(y_train))

    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        model = _fit_xgboost(X_train, y_train, calibration=cfg.model.calibration, seed=cfg.seed + idx)
        ensemble_models.append(model)

    importances = None
    try:
        importances = np.mean(
            [m.estimator.feature_importances_ for m in ensemble_models], axis=0
        )
    except AttributeError:
        pass

    return {
        "model_type": "xgboost",
        "feature_names": feature_names,
        "models": ensemble_models,
        "feature_importances": importances,
    }


def predict_probability(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
    prob_name: str = "p_xgb",
) -> tuple[xr.DataArray, xr.DataArray | None]:
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
