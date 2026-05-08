from __future__ import annotations

import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)

CLOUD_SCL_CLASSES = {3, 8, 9, 10, 11}


def _normalized_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    return (a - b) / (a + b)


def compute_s2_features(ds_s2: xr.Dataset) -> xr.Dataset | None:
    required = {"b03", "b08", "b11"}
    if not required.issubset(set(ds_s2.data_vars)):
        LOGGER.warning("S2 stack does not include all required bands for NDWI/MNDWI")
        return None

    b03 = ds_s2["b03"].astype("float32")
    b08 = ds_s2["b08"].astype("float32")
    b11 = ds_s2["b11"].astype("float32")

    valid_mask = xr.ones_like(b03, dtype=bool)
    if "scl" in ds_s2:
        scl = ds_s2["scl"]
        valid_mask = ~scl.isin(list(CLOUD_SCL_CLASSES))

    b03 = b03.where(valid_mask)
    b08 = b08.where(valid_mask)
    b11 = b11.where(valid_mask)

    ndwi = _normalized_difference(b03, b08).median(dim="time", skipna=True)
    mndwi = _normalized_difference(b03, b11).median(dim="time", skipna=True)
    valid_count = valid_mask.sum(dim="time")

    features = xr.Dataset(
        {
            "ndwi": ndwi.astype("float32"),
            "mndwi": mndwi.astype("float32"),
            "s2_valid_count": valid_count.astype("float32"),
        }
    )
    return features
