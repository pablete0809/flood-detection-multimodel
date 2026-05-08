from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import rioxarray
import xarray as xr

LOGGER = logging.getLogger(__name__)


def _empty_labels(template: xr.DataArray) -> xr.DataArray:
    labels = xr.full_like(template, -1, dtype="int8")
    labels.name = "labels"
    return labels


def load_external_label(
    label_path: str,
    template: xr.DataArray,
) -> xr.DataArray | None:
    if not label_path:
        return None
    path = Path(label_path)
    if not path.exists():
        LOGGER.warning("External label not found: %s", path)
        return None

    if path.suffix.lower() in {".tif", ".tiff"}:
        da = rioxarray.open_rasterio(path, masked=True).squeeze(drop=True)
        da = da.rio.reproject_match(template)
        bin_da = xr.where(da > 0.5, 1, 0).astype("int8")
        bin_da.name = "external_label"
        return bin_da

    gdf = gpd.read_file(path)
    if gdf.empty:
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(template.rio.crs)
    shapes = [(geom, 1) for geom in gdf.geometry]
    arr = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(template.sizes["y"], template.sizes["x"]),
        transform=template.rio.transform(),
        fill=0,
        dtype="uint8",
    )
    da = xr.DataArray(arr, coords=template.coords, dims=template.dims).astype("int8")
    da = da.rio.write_crs(template.rio.crs)
    da.name = "external_label"
    return da


def generate_water_pseudolabels(
    p_water_initial: xr.DataArray,
    freq_agua_hist: xr.DataArray,
    slope: xr.DataArray | None,
    external_label: xr.DataArray | None = None,
) -> xr.DataArray:
    labels = _empty_labels(p_water_initial)
    slope_ok = xr.ones_like(p_water_initial, dtype=bool)
    if slope is not None:
        slope_ok = slope < 8.0

    permanent = (freq_agua_hist > 0.6) & slope_ok
    high_conf_water = (p_water_initial > 0.8) | permanent
    high_conf_non_water = (p_water_initial < 0.2) & (freq_agua_hist < 0.25)

    labels = xr.where(high_conf_non_water, 0, labels)
    labels = xr.where(high_conf_water, 1, labels)

    if external_label is not None:
        labels = xr.where(external_label == 1, 1, labels)
        labels = xr.where(external_label == 0, 0, labels)

    labels = labels.astype("int8")
    labels.name = "water_label"
    return labels


def generate_flood_pseudolabels(
    p_water_initial: xr.DataArray,
    freq_agua_hist: xr.DataArray,
    slope: xr.DataArray | None,
    dist_to_permwater: xr.DataArray | None,
    external_label: xr.DataArray | None = None,
) -> xr.DataArray:
    labels = _empty_labels(p_water_initial)

    slope_ok = xr.ones_like(p_water_initial, dtype=bool)
    if slope is not None:
        slope_ok = slope < 8.0

    dist_ok = xr.ones_like(p_water_initial, dtype=bool)
    if dist_to_permwater is not None:
        dist_ok = dist_to_permwater < 2000.0

    flood_pos = (
        (p_water_initial > 0.6)
        & (freq_agua_hist < 0.2)
        & slope_ok
        & dist_ok
    )
    flood_neg = (freq_agua_hist > 0.5) | (p_water_initial < 0.2)

    labels = xr.where(flood_neg, 0, labels)
    labels = xr.where(flood_pos, 1, labels)

    if external_label is not None:
        labels = xr.where(external_label == 1, 1, labels)
        labels = xr.where(external_label == 0, 0, labels)

    labels = labels.astype("int8")
    labels.name = "flood_label"
    return labels


def flatten_samples(
    feature_ds: xr.Dataset,
    label_da: xr.DataArray,
    feature_names: list[str],
    extra_mask: xr.DataArray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    valid_features = [name for name in feature_names if name in feature_ds]
    if not valid_features:
        raise ValueError("No valid feature names found in dataset")

    x_arrays = [feature_ds[name].values.reshape(-1) for name in valid_features]
    X = np.stack(x_arrays, axis=1)
    y = label_da.values.reshape(-1).astype(np.int16)

    mask = np.isfinite(X).all(axis=1) & (y >= 0)
    if extra_mask is not None:
        mask = mask & (extra_mask.values.reshape(-1) > 0)
    return X[mask], y[mask]
