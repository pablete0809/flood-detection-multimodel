from __future__ import annotations

import logging
from dataclasses import dataclass
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

WATER_FEATURE_CANDIDATES = [
    "vv_db",
    "vh_db",
    "ratio_db",
    "delta_vv",
    "delta_vh",
    "slope",
    "ndwi",
    "mndwi",
    "freq_agua_hist",
]


@dataclass
class ThresholdChangeWaterModel:
    vv_thr: float
    vh_thr: float

    def predict_proba(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        idx = {name: i for i, name in enumerate(feature_names)}
        dvv = X[:, idx["delta_vv"]]
        dvh = X[:, idx["delta_vh"]]
        z = (self.vv_thr - dvv) + (self.vh_thr - dvh)
        if "ndwi" in idx:
            z = z + 1.25 * np.nan_to_num(X[:, idx["ndwi"]], nan=0.0)
        if "mndwi" in idx:
            z = z + 1.0 * np.nan_to_num(X[:, idx["mndwi"]], nan=0.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


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
        raise ValueError("Water model requires at least two classes")
    min_count = int(counts.min())
    cv = 3 if min_count >= 3 else 2
    method = calibration
    if method == "isotonic" and min_count < 30:
        method = "sigmoid"
        LOGGER.warning("Few samples for isotonic calibration; falling back to sigmoid.")
    model = CalibratedClassifierCV(base, method=method, cv=cv)
    model.fit(X, y)
    return model


def train_water_model_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray]],
    cfg: AppConfig,
) -> dict[str, Any]:
    model_type = cfg.model.water_model
    feature_names = [
        f
        for f in WATER_FEATURE_CANDIDATES
        if any(f in ds.data_vars for ds, _ in train_sets)
    ]
    if "delta_vv" not in feature_names or "delta_vh" not in feature_names:
        raise ValueError("Water model requires delta_vv and delta_vh features")

    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
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
    LOGGER.info("Water training samples: %s", len(y_train))

    if model_type == "threshold_change":
        model = ThresholdChangeWaterModel(
            vv_thr=cfg.model.threshold_change.vv_db,
            vh_thr=cfg.model.threshold_change.vh_db,
        )
        bundle = {
            "model_type": model_type,
            "feature_names": feature_names,
            "models": [model],
        }
        return bundle

    ensemble_models = []
    for idx in range(cfg.model.ensemble_n):
        seed = cfg.seed + idx
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
        "model_type": "logreg",
        "feature_names": feature_names,
        "models": ensemble_models,
    }


def save_model_bundle(bundle: dict[str, Any], out_path: str) -> None:
    joblib.dump(bundle, out_path)


def load_model_bundle(path: str) -> dict[str, Any]:
    return joblib.load(path)


def _dataset_to_matrix(
    feature_ds: xr.Dataset, feature_names: list[str]
) -> tuple[np.ndarray, xr.DataArray]:
    template_name = feature_names[0]
    template = feature_ds[template_name]
    data = [feature_ds[name].values.reshape(-1) for name in feature_names]
    X = np.stack(data, axis=1).astype(np.float32)
    return X, template


def predict_water_probability(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray | None]:
    feature_names = bundle["feature_names"]
    missing = [name for name in feature_names if name not in feature_ds]
    if missing:
        raise ValueError(f"Missing required features for water prediction: {missing}")
    X, template = _dataset_to_matrix(feature_ds, feature_names)

    valid = np.isfinite(X).all(axis=1)
    probs = np.full((len(bundle["models"]), X.shape[0]), np.nan, dtype=np.float32)
    for i, model in enumerate(bundle["models"]):
        if isinstance(model, ThresholdChangeWaterModel):
            p = model.predict_proba(X[valid], feature_names)[:, 1]
        else:
            p = model.predict_proba(X[valid])[:, 1]
        probs[i, valid] = p.astype(np.float32)

    p_mean = np.nanmean(probs, axis=0)
    p_std = np.nanstd(probs, axis=0) if probs.shape[0] > 1 else None

    mean_da = xr.DataArray(
        p_mean.reshape(template.shape),
        coords=template.coords,
        dims=template.dims,
        name="p_agua",
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
            name="p_agua_std",
        ).astype("float32")
        if template_crs is not None:
            std_da = std_da.rio.write_crs(template_crs)
    return mean_da, std_da
