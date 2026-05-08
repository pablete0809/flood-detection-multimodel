from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

LOGGER = logging.getLogger(__name__)


def load_dem_features(
    dem_path: str | Path,
    template: xr.DataArray,
) -> xr.Dataset | None:
    dem_path_obj = Path(dem_path)
    if not dem_path_obj.exists():
        LOGGER.warning("DEM path not found: %s", dem_path_obj)
        return None

    dem = rioxarray.open_rasterio(dem_path_obj, masked=True).squeeze(drop=True)
    dem = dem.rio.reproject_match(template)
    dem = dem.astype("float32")

    res_x, res_y = dem.rio.resolution()
    dx = abs(float(res_x)) if res_x else 10.0
    dy = abs(float(res_y)) if res_y else 10.0

    arr = dem.values.astype(np.float32)
    grad_y, grad_x = np.gradient(arr, dy, dx)
    slope = np.degrees(np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))).astype("float32")

    slope_da = xr.DataArray(
        slope,
        coords=dem.coords,
        dims=dem.dims,
        name="slope",
    ).rio.write_crs(dem.rio.crs)

    out = xr.Dataset({"elevation": dem.rename("elevation"), "slope": slope_da})
    return out


def compute_distance_to_permwater(
    freq_agua_hist: xr.DataArray,
    threshold: float = 0.6,
    resolution_m: float = 10.0,
) -> xr.DataArray:
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore
    except Exception:  # noqa: BLE001
        LOGGER.warning("scipy not available. dist_to_permwater will be NaN.")
        nan_arr = xr.full_like(freq_agua_hist, np.nan, dtype="float32")
        nan_arr.name = "dist_to_permwater"
        return nan_arr

    perm = (freq_agua_hist >= threshold).fillna(False).values.astype(bool)
    dist_px = distance_transform_edt(~perm).astype("float32")
    dist_m = dist_px * float(resolution_m)
    da = xr.DataArray(
        dist_m,
        coords=freq_agua_hist.coords,
        dims=freq_agua_hist.dims,
        name="dist_to_permwater",
    ).rio.write_crs(freq_agua_hist.rio.crs)
    return da
