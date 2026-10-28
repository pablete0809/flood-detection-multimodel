from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import flatten_samples

LOGGER = logging.getLogger(__name__)

FLOOD_FEATURE_CANDIDATES = [
    "delta_vv",
    "delta_vh",
    "vv_db",
    "vh_db",
    "slope",
    "dist_to_permwater",
    "ndwi",
    "mndwi",
]


def _fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    calibration: str,
    c_value: float,
    max_iter: int,
    seed: int,
):
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )),
    ])
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Flood model requires at least two classes")
    min_count = int(counts.min())
    cv = 3 if min_count >= 3 else 2
    method = calibration
    if method == "isotonic" and min_count < 30:
        method = "sigmoid"
        LOGGER.warning("Few samples for isotonic calibration; falling back to sigmoid.")
    model = CalibratedClassifierCV(base, method=method, cv=cv)
    model.fit(X, y)
    return model


def _fit_xgboost_if_available(
    X: np.ndarray,
    y: np.ndarray,
    calibration: str,
    seed: int,
):
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception:  # noqa: BLE001
        LOGGER.warning("xgboost not available; fallback to logistic regression.")
        return _fit_logreg(
            X,
            y,
            calibration=calibration,
            c_value=1.0,
            max_iter=500,
            seed=seed,
        )

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
        raise ValueError("Flood model requires at least two classes")
    min_count = int(counts.min())
    cv = 3 if min_count >= 3 else 2
    method = calibration
    if method == "isotonic" and min_count < 30:
        method = "sigmoid"
    model = CalibratedClassifierCV(base, method=method, cv=cv)
    model.fit(X, y)
    return model


def train_flood_model_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    feature_names = [
        f
        for f in FLOOD_FEATURE_CANDIDATES
        if any(f in ds.data_vars for ds, _, _ in train_sets)
    ]
    required = {"delta_vv", "delta_vh"}
    if not required.issubset(set(feature_names)):
        raise ValueError(
            f"Flood model requires features {sorted(required)}. Found: {feature_names}"
        )

    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for ds, flood_labels, _p_water in train_sets:
        X, y = flatten_samples(
            ds,
            flood_labels,
            feature_names=feature_names,
        )
        if len(y) == 0:
            continue
        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise ValueError("No flood training samples after filtering")

    X_train = np.vstack(X_all)
    y_train = np.concatenate(y_all)
    LOGGER.info("Flood training samples: %s", len(y_train))

    model_type = cfg.model.flood_given_water_model
    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        seed = cfg.seed + idx
        if model_type == "xgboost_if_available":
            model = _fit_xgboost_if_available(
                X_train,
                y_train,
                calibration=cfg.model.calibration,
                seed=seed,
            )
        else:
            model = _fit_logreg(
                X_train,
                y_train,
                calibration=cfg.model.calibration,
                c_value=cfg.model.logreg.c,
                max_iter=cfg.model.logreg.max_iter,
                seed=seed,
            )
        ensemble_models.append(model)

    return {
        "model_type": model_type,
        "feature_names": feature_names,
        "models": ensemble_models,
    }


def save_model_bundle(bundle: dict[str, Any], out_path: str) -> None:
    joblib.dump(bundle, out_path)


def load_model_bundle(path: str) -> dict[str, Any]:
    return joblib.load(path)


def predict_flood_probability(
    feature_ds: xr.Dataset,
    p_water: xr.DataArray,
    bundle: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray | None]:
    feature_names = bundle["feature_names"]
    missing = [name for name in feature_names if name not in feature_ds]
    if missing:
        raise ValueError(f"Missing required features for flood prediction: {missing}")

    # The flood head is trained on every labelled pixel (not only on water), so it
    # predicts everywhere; p_water only provides the output grid. Masking by p_water
    # here made sklearn models be scored on a different pixel set than the CNNs.
    template = p_water
    X = np.stack([feature_ds[name].values.reshape(-1) for name in feature_names], axis=1)
    valid = np.isfinite(X).all(axis=1)

    probs = np.full((len(bundle["models"]), X.shape[0]), np.nan, dtype=np.float32)
    for i, model in enumerate(bundle["models"]):
        p = model.predict_proba(X[valid])[:, 1]
        probs[i, valid] = p.astype(np.float32)

    p_mean = np.nanmean(probs, axis=0)
    p_std = np.nanstd(probs, axis=0) if probs.shape[0] > 1 else None

    mean_da = xr.DataArray(
        p_mean.reshape(template.shape),
        coords=template.coords,
        dims=template.dims,
        name="p_inundacion_dado_agua",
    ).astype("float32")
    template_crs = template.rio.crs
    if template_crs is not None:
        mean_da = mean_da.rio.write_crs(template_crs)

    std_da = None
    if p_std is not None:
        std_da = xr.DataArray(
            p_std.reshape(template.shape),
            coords=template.coords,
            dims=template.dims,
            name="p_inundacion_dado_agua_std",
        ).astype("float32")
        if template_crs is not None:
            std_da = std_da.rio.write_crs(template_crs)

    return mean_da, std_da
