from __future__ import annotations

import logging
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pystac
import rasterio
import rioxarray  # noqa: F401
import stackstac
import xarray as xr
from rasterio.enums import Resampling

from floodprob.utils.geospatial import aoi_bbox_wgs84, clip_dataset_to_aoi
from floodprob.utils.io import ensure_dir, hash_payload, with_retries

LOGGER = logging.getLogger(__name__)

S1_ASSET_CANDIDATES = {
    "vv": ["vv", "VV", "s1_vv", "sigma0_vv", "measurement_vv"],
    "vh": ["vh", "VH", "s1_vh", "sigma0_vh", "measurement_vh"],
}

S2_ASSET_CANDIDATES = {
    "b03": ["B03", "b03", "green", "sr_band3"],
    "b08": ["B08", "b08", "nir", "sr_band8"],
    "b11": ["B11", "b11", "swir16", "sr_band11"],
    "scl": ["SCL", "scl", "scene_classification", "classification"],
}


def _asset_has_valid_crs(href: str) -> bool:
    try:
        with rasterio.Env(
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2,.img,.nc",
            GDAL_HTTP_UNSAFESSL="YES",
        ):
            with rasterio.open(href) as src:
                return src.crs is not None
    except Exception:  # noqa: BLE001
        return False


def _filter_items_by_crs(
    items: list[pystac.Item],
    asset_priority: list[str],
    skip_crs_check: bool = False,
) -> list[pystac.Item]:
    if not items:
        return []
    if skip_crs_check:
        LOGGER.info(
            "Skipping CRS asset-check for %s items (epsg provided via stackstac).", len(items)
        )
        return items
    kept: list[pystac.Item] = []
    dropped = 0
    for item in items:
        ok = False
        for asset_name in asset_priority:
            asset = item.assets.get(asset_name)
            if asset is None or not asset.href:
                continue
            if _asset_has_valid_crs(asset.href):
                ok = True
                break
        if ok:
            kept.append(item)
        else:
            dropped += 1
    if dropped > 0:
        LOGGER.warning("Dropped %s items with unreadable/missing CRS in assets.", dropped)
    return kept


def _find_asset_key(item: pystac.Item, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in item.assets:
            return candidate
    low_map = {k.lower(): k for k in item.assets.keys()}
    for candidate in candidates:
        key = low_map.get(candidate.lower())
        if key:
            return key
    return None


def _prepare_items_for_sensor(
    items: list[pystac.Item], sensor: str
) -> tuple[list[pystac.Item], list[str]]:
    if sensor not in {"s1", "s2"}:
        raise ValueError(f"Unsupported sensor: {sensor}")

    candidates = S1_ASSET_CANDIDATES if sensor == "s1" else S2_ASSET_CANDIDATES
    required = ["vv", "vh"] if sensor == "s1" else ["b03", "b08", "b11"]
    optional = [] if sensor == "s1" else ["scl"]

    valid_items: list[pystac.Item] = []
    per_item_present: list[set[str]] = []
    for item in items:
        present: set[str] = set()
        ok = True
        for canonical in required:
            key = _find_asset_key(item, candidates[canonical])
            if key is None:
                ok = False
                break
            item.assets[canonical] = item.assets[key]
            present.add(canonical)
        if not ok:
            continue
        for canonical in optional:
            key = _find_asset_key(item, candidates[canonical])
            if key is not None:
                item.assets[canonical] = item.assets[key]
                present.add(canonical)
        valid_items.append(item)
        per_item_present.append(present)

    if not valid_items:
        return [], []

    valid_items = _filter_items_by_crs(valid_items, required, skip_crs_check=True)
    if not valid_items:
        return [], []

    per_item_present = []
    for item in valid_items:
        present = {a for a in required + optional if a in item.assets}
        per_item_present.append(present)

    common_assets = set.intersection(*per_item_present)
    ordered_assets = [a for a in required if a in common_assets]
    ordered_assets.extend([a for a in optional if a in common_assets])
    if any(a not in ordered_assets for a in required):
        return [], []
    return valid_items, ordered_assets


def _stack_to_dataset(
    items: list[pystac.Item],
    assets: list[str],
    aoi_gdf: gpd.GeoDataFrame,
    epsg: int,
    resolution_m: int,
) -> xr.Dataset:
    bbox = aoi_bbox_wgs84(aoi_gdf)
    with rasterio.Env(CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2,.img,.nc"):
        arr = stackstac.stack(
            items,
            assets=assets,
            epsg=epsg,
            resolution=resolution_m,
            bounds_latlon=bbox,
            fill_value=np.float32(np.nan),
            dtype="float32",
            rescale=False,
            errors_as_nodata=(Exception,),
            resampling=Resampling.nearest,
            chunksize=1024,
        )

    ds = arr.to_dataset(dim="band")
    ds = ds.rio.write_crs(epsg)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
    ds = clip_dataset_to_aoi(ds, aoi_gdf)
    return ds


def _sanitize_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    keep_coords = {"time", "x", "y", "spatial_ref"}
    drop_coords = [name for name in ds.coords if name not in keep_coords]
    if drop_coords:
        ds = ds.drop_vars(drop_coords, errors="ignore")

    for name in list(ds.coords):
        if ds[name].dtype == object:
            ds = ds.drop_vars(name, errors="ignore")

    ds.attrs = {}
    for name in ds.data_vars:
        ds[name].attrs = {}
    for name in ds.coords:
        ds[name].attrs = {}
    return ds


def build_or_load_stack(
    items: list[pystac.Item],
    sensor: str,
    aoi_gdf: gpd.GeoDataFrame,
    epsg: int,
    resolution_m: int,
    cache_dir: str | Path,
    cache_prefix: str,
) -> Path:
    prepared_items, assets = _prepare_items_for_sensor(items, sensor=sensor)
    if not prepared_items:
        raise ValueError(f"No valid items/assets found for sensor={sensor}")

    cache_base = ensure_dir(Path(cache_dir) / "stacks")
    payload = {
        "sensor": sensor,
        "item_ids": [it.id for it in prepared_items],
        "assets": assets,
        "epsg": epsg,
        "resolution_m": resolution_m,
        "zarr_version": 2,
        "aoi_bounds": [float(v) for v in aoi_gdf.to_crs(4326).total_bounds],
    }
    key = hash_payload(payload)
    cache_path = cache_base / f"{cache_prefix}_{key}.zarr"
    if cache_path.exists():
        try:
            test = xr.open_zarr(cache_path, consolidated=True)
            _ = list(test.data_vars)
            LOGGER.info("Using cached stack: %s", cache_path)
            return cache_path
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Invalid cache detected (%s). Rebuilding %s", exc, cache_path)
            shutil.rmtree(cache_path, ignore_errors=True)

    LOGGER.info(
        "Building stack | sensor=%s items=%s assets=%s", sensor, len(prepared_items), assets
    )
    ds = with_retries(
        _stack_to_dataset,
        prepared_items,
        assets,
        aoi_gdf,
        epsg,
        resolution_m,
        retries=3,
        sleep_s=2.0,
        label=f"stack_{sensor}",
    )
    ds = _sanitize_for_zarr(ds)
    ds = ds.chunk({"time": 1, "y": 1024, "x": 1024})
    with_retries(
        ds.to_zarr,
        cache_path,
        mode="w",
        consolidated=True,
        zarr_version=2,
        retries=3,
        sleep_s=2.0,
        label=f"save_stack_{sensor}",
    )
    LOGGER.info("Saved stack to %s", cache_path)
    return cache_path


def open_stack(path: str | Path, fallback_crs: str | None = None) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=True)
    if fallback_crs is not None and ds.rio.crs is None:
        ds = ds.rio.write_crs(fallback_crs)
    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    return ds
