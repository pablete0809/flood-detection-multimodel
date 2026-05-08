from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS

LOGGER = logging.getLogger(__name__)


def read_aoi(aoi_geojson: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(aoi_geojson)
    if gdf.empty:
        raise ValueError(f"AOI is empty: {aoi_geojson}")
    if gdf.crs is None:
        LOGGER.warning("AOI CRS is missing. Assuming EPSG:4326.")
        gdf = gdf.set_crs(4326)
    return gdf


def aoi_geometry_wgs84(aoi_gdf: gpd.GeoDataFrame) -> dict:
    gdf = aoi_gdf.to_crs(4326)
    return gdf.geometry.iloc[0].__geo_interface__


def aoi_bbox_wgs84(aoi_gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    gdf = aoi_gdf.to_crs(4326)
    minx, miny, maxx, maxy = gdf.total_bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def parse_epsg(crs_text: str) -> int:
    crs_obj = CRS.from_user_input(crs_text)
    epsg = crs_obj.to_epsg()
    if epsg is None:
        raise ValueError(f"Could not resolve EPSG code from {crs_text}")
    return int(epsg)


def clip_dataset_to_aoi(ds: xr.Dataset, aoi_gdf: gpd.GeoDataFrame) -> xr.Dataset:
    if ds.rio.crs is None:
        raise ValueError("Dataset has no CRS. Cannot clip to AOI.")
    aoi_local = aoi_gdf.to_crs(ds.rio.crs)
    clipped = ds.rio.clip(aoi_local.geometry.values, aoi_local.crs, drop=True)
    return clipped


def align_dataset_to_template(ds: xr.Dataset, template: xr.DataArray) -> xr.Dataset:
    aligned_vars: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        aligned_vars[name] = da.rio.reproject_match(template)
    return xr.Dataset(aligned_vars)


def ensure_xy_crs(da: xr.DataArray, crs: str | int | None) -> xr.DataArray:
    if da.rio.crs is None:
        if crs is None:
            raise ValueError("DataArray CRS is missing and no fallback CRS was provided.")
        da = da.rio.write_crs(crs)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    return da


def save_geotiff(
    da: xr.DataArray,
    out_path: str | Path,
    nodata: float = np.nan,
    compress: str = "deflate",
    fallback_crs: str | int | None = None,
) -> None:
    out = da.astype("float32")
    # Prioritise CRS already in the DataArray; use caller-provided fallback if missing.
    resolved_crs = da.rio.crs if da.rio.crs is not None else fallback_crs
    out = ensure_xy_crs(out, resolved_crs)
    out = out.rio.write_nodata(nodata, inplace=False)
    out.rio.to_raster(
        Path(out_path),
        compress=compress,
        tiled=True,
        dtype="float32",
        predictor=2,
    )
