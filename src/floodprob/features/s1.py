from __future__ import annotations

import numpy as np
import xarray as xr


def to_db(backscatter: xr.DataArray) -> xr.DataArray:
    valid = backscatter.where(backscatter > 0)
    # Use xarray median (dask-safe) instead of np.nanmedian(backscatter.values)
    # which triggers a numpy.ma IndexError with chunked dask arrays.
    try:
        median_val = float(backscatter.median(skipna=True).compute().item())
    except Exception:
        median_val = float("nan")
    if np.isfinite(median_val) and median_val <= 1.0:
        out = 10.0 * np.log10(valid)
    else:
        out = backscatter
    return out.astype("float32")


def compute_s1_baseline(ds_s1: xr.Dataset) -> xr.Dataset:
    if "vv" not in ds_s1 or "vh" not in ds_s1:
        raise ValueError("S1 dataset must include vv and vh variables")

    vv_db = to_db(ds_s1["vv"])
    vh_db = to_db(ds_s1["vh"])
    ratio_db = vv_db - vh_db

    baseline = xr.Dataset(
        {
            "vv_db_base": vv_db.median(dim="time", skipna=True),
            "vh_db_base": vh_db.median(dim="time", skipna=True),
            "ratio_db_base": ratio_db.median(dim="time", skipna=True),
        }
    )
    return baseline


def compute_s1_target_features(ds_s1_target: xr.Dataset, baseline: xr.Dataset) -> xr.Dataset:
    vv_db = to_db(ds_s1_target["vv"]).median(dim="time", skipna=True)
    vh_db = to_db(ds_s1_target["vh"]).median(dim="time", skipna=True)
    ratio_db = vv_db - vh_db

    feat = xr.Dataset(
        {
            "vv_db": vv_db,
            "vh_db": vh_db,
            "ratio_db": ratio_db,
            "delta_vv": vv_db - baseline["vv_db_base"],
            "delta_vh": vh_db - baseline["vh_db_base"],
        }
    )
    return feat


def threshold_water_probability(
    features: xr.Dataset,
    vv_threshold_db: float,
    vh_threshold_db: float,
) -> xr.DataArray:
    dvv = features["delta_vv"]
    dvh = features["delta_vh"]
    z = (vv_threshold_db - dvv) + (vh_threshold_db - dvh)
    if "ratio_db" in features:
        z = z + 0.15 * (-features["ratio_db"])
    if "ndwi" in features:
        z = z + 1.25 * features["ndwi"].fillna(0.0)
    if "mndwi" in features:
        z = z + 1.0 * features["mndwi"].fillna(0.0)
    prob = 1.0 / (1.0 + np.exp(-z))
    prob = prob.clip(0.0, 1.0).astype("float32")
    prob.name = "p_water_initial"
    return prob
