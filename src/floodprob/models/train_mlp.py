from __future__ import annotations

"""MLP (sklearn) model for water and flood heads.

Uses sklearn.neural_network.MLPClassifier — no GPU required, good for tabular data.
Three hidden layers by default (configurable via model.compare.mlp.hidden_layer_sizes).
Wrapped in CalibratedClassifierCV to ensure well-calibrated probability outputs.
"""

import logging
from typing import Any

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import flatten_samples
from floodprob.models.train_water import (
    WATER_FEATURE_CANDIDATES,
    _dataset_to_matrix,
)
from floodprob.models.train_flood_given_water import FLOOD_FEATURE_CANDIDATES

LOGGER = logging.getLogger(__name__)


def _fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    calibration: str,
    hidden_layer_sizes: list[int],
    max_iter: int,
    alpha: float,
    seed: int,
) -> CalibratedClassifierCV:
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            activation="relu",
            solver="adam",
            alpha=alpha,
            max_iter=max_iter,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )),
    ])
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("MLP model requires at least two classes")
    min_count = int(counts.min())
    cv = 3 if min_count >= 3 else 2
    method = calibration
    if method == "isotonic" and min_count < 30:
        method = "sigmoid"
        LOGGER.warning("Few samples for isotonic calibration; falling back to sigmoid (MLP).")
    model = CalibratedClassifierCV(base, method=method, cv=cv)
    model.fit(X, y)
    return model


def train_water_mlp_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    feature_names = [
        f for f in WATER_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _ in train_sets)
    ]
    if "delta_vv" not in feature_names or "delta_vh" not in feature_names:
        raise ValueError("MLP water model requires delta_vv and delta_vh")

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
    LOGGER.info("MLP Water training samples: %s", len(y_train))

    mlp_cfg = cfg.model.compare.mlp
    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        model = _fit_mlp(
            X_train, y_train,
            calibration=cfg.model.calibration,
            hidden_layer_sizes=mlp_cfg.hidden_layer_sizes,
            max_iter=mlp_cfg.max_iter,
            alpha=mlp_cfg.alpha,
            seed=cfg.seed + idx,
        )
        ensemble_models.append(model)

    return {
        "model_type": "mlp",
        "feature_names": feature_names,
        "models": ensemble_models,
        "feature_importances": None,  # MLPs don't have native importances
    }


def train_flood_mlp_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    feature_names = [
        f for f in FLOOD_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _, _ in train_sets)
    ]
    required = {"delta_vv", "delta_vh"}
    if not required.issubset(set(feature_names)):
        raise ValueError(f"MLP flood model requires features {sorted(required)}")

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
    LOGGER.info("MLP Flood training samples: %s", len(y_train))

    mlp_cfg = cfg.model.compare.mlp
    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        model = _fit_mlp(
            X_train, y_train,
            calibration=cfg.model.calibration,
            hidden_layer_sizes=mlp_cfg.hidden_layer_sizes,
            max_iter=mlp_cfg.max_iter,
            alpha=mlp_cfg.alpha,
            seed=cfg.seed + idx,
        )
        ensemble_models.append(model)

    return {
        "model_type": "mlp",
        "feature_names": feature_names,
        "models": ensemble_models,
        "feature_importances": None,
    }


def predict_probability(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
    prob_name: str = "p_mlp",
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
