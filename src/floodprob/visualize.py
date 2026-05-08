from __future__ import annotations

import logging
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from rasterio.warp import transform_bounds

from floodprob.config import AppConfig
from floodprob.utils.io import build_run_dirs, read_json

LOGGER = logging.getLogger(__name__)


def _to_rgba(data: np.ndarray, cmap_name: str) -> np.ndarray:
    mask = ~np.isfinite(data)
    normalized = np.clip(data, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    rgba[mask, 3] = 0
    return rgba


def _raster_bounds_wgs84(da) -> tuple[float, float, float, float]:
    left, bottom, right, top = da.rio.bounds()
    crs = da.rio.crs
    if crs is None:
        raise ValueError("Raster has no CRS. Cannot build folium overlay bounds.")
    if str(crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        return (left, bottom, right, top)
    return transform_bounds(crs, "EPSG:4326", left, bottom, right, top)


def run_folium_preview(cfg: AppConfig, key: str | None = None) -> Path:
    paths = build_run_dirs(cfg.outputs.out_dir, cfg.run_name)
    predict_manifest_path = paths["metadata"] / "manifest_predict.json"
    if not predict_manifest_path.exists():
        raise FileNotFoundError(
            f"Prediction manifest not found: {predict_manifest_path}. Run `floodprob predict` first."
        )
    manifest = read_json(predict_manifest_path)
    predictions = manifest.get("predictions", [])
    if not predictions:
        raise ValueError("No predictions available in manifest_predict.json")

    record = None
    if key is None:
        record = predictions[0]
    else:
        for pred in predictions:
            if pred.get("key") == key:
                record = pred
                break
    if record is None:
        available = [p.get("key") for p in predictions]
        raise ValueError(f"Key '{key}' not found. Available keys: {available}")

    p_agua = rioxarray.open_rasterio(record["p_agua"], masked=True).squeeze(drop=True)
    p_flood_gw = rioxarray.open_rasterio(
        record["p_inundacion_dado_agua"], masked=True
    ).squeeze(drop=True)
    p_flood = rioxarray.open_rasterio(record["p_inundacion"], masked=True).squeeze(drop=True)

    bounds = _raster_bounds_wgs84(p_agua)
    west, south, east, north = bounds
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    map_obj = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer("CartoDB positron", name="CartoDB Positron").add_to(map_obj)

    overlay_bounds = [[south, west], [north, east]]
    layers = [
        ("P_agua", p_agua.values.astype("float32"), "Blues", True),
        (
            "P_inundacion_dado_agua",
            p_flood_gw.values.astype("float32"),
            "Oranges",
            True,
        ),
        ("P_inundacion", p_flood.values.astype("float32"), "Reds", True),
    ]
    for name, arr, cmap, shown in layers:
        rgba = _to_rgba(arr, cmap_name=cmap)
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=overlay_bounds,
            opacity=0.72,
            name=name,
            interactive=True,
            cross_origin=False,
            zindex=2,
            show=shown,
        ).add_to(map_obj)

    folium.LayerControl(collapsed=False).add_to(map_obj)

    out_path = paths["evaluation"] / f"folium_preview_{record['key']}.html"
    map_obj.save(str(out_path))
    LOGGER.info("Folium preview saved to %s", out_path)
    return out_path
