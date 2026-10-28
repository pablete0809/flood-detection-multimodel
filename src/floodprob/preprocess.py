from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from floodprob.config import AppConfig
from floodprob.features.dem import compute_distance_to_permwater, load_dem_features
from floodprob.features.s1 import (
    compute_s1_baseline,
    compute_s1_target_features,
    threshold_water_probability,
)
from floodprob.features.s2 import compute_s2_features
from floodprob.ingest.stack import build_or_load_stack, open_stack
from floodprob.ingest.stac_search import (
    collection_exists,
    create_stac_client,
    list_collection_ids,
    sample_item_asset_keys,
    search_sensor_items,
)
from floodprob.labels.pseudolabels import (
    build_ground_truth,
    generate_flood_pseudolabels,
    generate_water_pseudolabels,
    load_external_label,
)
from floodprob.utils.geospatial import (
    align_dataset_to_template,
    aoi_geometry_wgs84,
    parse_epsg,
    read_aoi,
    save_geotiff,
)
from floodprob.utils.io import build_run_dirs, read_json, window_key, write_json

LOGGER = logging.getLogger(__name__)


def _run_paths(cfg: AppConfig) -> dict[str, Path]:
    return build_run_dirs(cfg.outputs.out_dir, cfg.run_name)


def run_stac_check(cfg: AppConfig) -> dict[str, Any]:
    paths = _run_paths(cfg)
    client = create_stac_client(cfg.stac)
    collections = list_collection_ids(client)
    aoi = read_aoi(cfg.aoi_geojson)
    aoi_geom = aoi_geometry_wgs84(aoi)

    out: dict[str, Any] = {
        "api_url": cfg.stac.api_url,
        "collection_count": len(collections),
        "configured_collections": {
            "s1": cfg.stac.collections.s1,
            "s2": cfg.stac.collections.s2,
        },
        "collection_exists": {
            "s1": collection_exists(client, cfg.stac.collections.s1),
            "s2": collection_exists(client, cfg.stac.collections.s2),
        },
        "search_probe": {},
    }

    probe_window = cfg.time.targets[0]
    for sensor, collection in (
        ("s1", cfg.stac.collections.s1),
        ("s2", cfg.stac.collections.s2),
    ):
        try:
            items = search_sensor_items(
                client=client,
                stac_cfg=cfg.stac,
                sensor=sensor,
                collection=collection,
                aoi_geojson=aoi_geom,
                window=probe_window,
                filters=cfg.filters,
            )
            out["search_probe"][sensor] = {
                "window": [str(probe_window[0]), str(probe_window[1])],
                "items_found": len(items),
                "sample_asset_keys": sample_item_asset_keys(items, n=2),
            }
        except Exception as exc:  # noqa: BLE001
            out["search_probe"][sensor] = {"error": str(exc)}

    report_path = paths["metadata"] / "stac_check.json"
    write_json(report_path, out)
    LOGGER.info("STAC check written to %s", report_path)
    return out


def _search_and_stack(
    cfg: AppConfig,
    aoi_geom: dict[str, Any],
    aoi_gdf,
    client,
    sensor: str,
    collection: str,
    window: tuple,
    cache_tag: str,
    relative_orbits: set[int] | None = None,
) -> str:
    items = search_sensor_items(
        client=client,
        stac_cfg=cfg.stac,
        sensor=sensor,
        collection=collection,
        aoi_geojson=aoi_geom,
        window=window,
        filters=cfg.filters,
    )
    if relative_orbits:
        n_before = len(items)
        items = [it for it in items if _relative_orbit(it) in relative_orbits]
        LOGGER.info(
            "%s: %d/%d items kept for relative orbits %s",
            cache_tag, len(items), n_before, sorted(relative_orbits),
        )
    if not items:
        return ""
    epsg = parse_epsg(cfg.crs_salida)
    cache_path = build_or_load_stack(
        items=items,
        sensor=sensor,
        aoi_gdf=aoi_gdf,
        epsg=epsg,
        resolution_m=cfg.resolution_m,
        cache_dir=cfg.outputs.cache_dir,
        cache_prefix=cache_tag,
    )
    return str(cache_path)


def _relative_orbit(item) -> int | None:
    value = item.properties.get("sat:relative_orbit")
    return int(value) if value is not None else None


def _target_relative_orbits(cfg: AppConfig, client, aoi_geom: dict[str, Any]) -> set[int] | None:
    """Órbitas relativas S1 de las ventanas de evento (None si no se pueden determinar)."""
    orbits: set[int] = set()
    for window in cfg.time.targets:
        items = search_sensor_items(
            client=client,
            stac_cfg=cfg.stac,
            sensor="s1",
            collection=cfg.stac.collections.s1,
            aoi_geojson=aoi_geom,
            window=window,
            filters=cfg.filters,
        )
        orbits.update(o for o in (_relative_orbit(it) for it in items) if o is not None)
    if not orbits:
        LOGGER.warning("S1 items have no sat:relative_orbit; baseline orbit filter disabled.")
        return None
    return orbits


def run_ingest(cfg: AppConfig) -> dict[str, Any]:
    paths = _run_paths(cfg)
    client = create_stac_client(cfg.stac)
    aoi = read_aoi(cfg.aoi_geojson)
    aoi_geom = aoi_geometry_wgs84(aoi)

    s1_orbits = None
    if cfg.filters.s1_match_target_orbit:
        s1_orbits = _target_relative_orbits(cfg, client, aoi_geom)

    historical_windows = cfg.time.historical_windows[: cfg.model.n_hist_windows]
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": cfg.run_name,
        "api_url": cfg.stac.api_url,
        "s1_relative_orbits": sorted(s1_orbits) if s1_orbits else None,
        "baseline": {
            "window": [str(cfg.time.baseline[0]), str(cfg.time.baseline[1])],
            "s1": "",
            "s2": "",
        },
        "targets": [],
        "historical": [],
    }

    base_key = window_key(cfg.time.baseline)
    baseline_s1 = _search_and_stack(
        cfg,
        aoi_geom,
        aoi,
        client,
        sensor="s1",
        collection=cfg.stac.collections.s1,
        window=cfg.time.baseline,
        cache_tag=f"s1_baseline_{base_key}",
        relative_orbits=s1_orbits,
    )
    if not baseline_s1:
        raise RuntimeError("No Sentinel-1 items found for baseline window.")
    manifest["baseline"]["s1"] = baseline_s1

    baseline_s2 = _search_and_stack(
        cfg,
        aoi_geom,
        aoi,
        client,
        sensor="s2",
        collection=cfg.stac.collections.s2,
        window=cfg.time.baseline,
        cache_tag=f"s2_baseline_{base_key}",
    )
    manifest["baseline"]["s2"] = baseline_s2

    for window in cfg.time.targets:
        key = window_key(window)
        target_record = {
            "window": [str(window[0]), str(window[1])],
            "key": key,
            "s1": "",
            "s2": "",
        }
        target_record["s1"] = _search_and_stack(
            cfg,
            aoi_geom,
            aoi,
            client,
            sensor="s1",
            collection=cfg.stac.collections.s1,
            window=window,
            cache_tag=f"s1_target_{key}",
        )
        if not target_record["s1"]:
            raise RuntimeError(f"No Sentinel-1 items found for target window {key}")
        target_record["s2"] = _search_and_stack(
            cfg,
            aoi_geom,
            aoi,
            client,
            sensor="s2",
            collection=cfg.stac.collections.s2,
            window=window,
            cache_tag=f"s2_target_{key}",
        )
        manifest["targets"].append(target_record)

    for window in historical_windows:
        key = window_key(window)
        hist_record = {
            "window": [str(window[0]), str(window[1])],
            "key": key,
            "s1": "",
            "s2": "",
        }
        hist_record["s1"] = _search_and_stack(
            cfg,
            aoi_geom,
            aoi,
            client,
            sensor="s1",
            collection=cfg.stac.collections.s1,
            window=window,
            cache_tag=f"s1_hist_{key}",
            relative_orbits=s1_orbits,
        )
        hist_record["s2"] = _search_and_stack(
            cfg,
            aoi_geom,
            aoi,
            client,
            sensor="s2",
            collection=cfg.stac.collections.s2,
            window=window,
            cache_tag=f"s2_hist_{key}",
        )
        manifest["historical"].append(hist_record)

    manifest_path = paths["metadata"] / "manifest_ingest.json"
    write_json(manifest_path, manifest)
    LOGGER.info("Ingest manifest written to %s", manifest_path)
    return manifest


def _load_ingest_manifest(cfg: AppConfig) -> dict[str, Any]:
    paths = _run_paths(cfg)
    manifest_path = paths["metadata"] / "manifest_ingest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Ingest manifest not found: {manifest_path}. Run `floodprob ingest` first."
        )
    return read_json(manifest_path)


def run_preprocess(cfg: AppConfig) -> dict[str, Any]:
    paths = _run_paths(cfg)
    ingest_manifest = _load_ingest_manifest(cfg)

    baseline_s1_path = ingest_manifest["baseline"]["s1"]
    baseline_s1 = open_stack(baseline_s1_path, fallback_crs=cfg.crs_salida)
    baseline_feat = compute_s1_baseline(baseline_s1)
    baseline_template = baseline_feat["vv_db_base"]

    dem_features = None
    if cfg.dem.path:
        dem_features = load_dem_features(cfg.dem.path, baseline_template)

    hist_probs: list[xr.DataArray] = []
    for hist_record in ingest_manifest.get("historical", []):
        s1_path = hist_record.get("s1", "")
        if not s1_path:
            continue
        hist_ds = open_stack(s1_path, fallback_crs=cfg.crs_salida)
        hist_feat = compute_s1_target_features(hist_ds, baseline_feat)
        hist_p = threshold_water_probability(
            hist_feat,
            vv_threshold_db=cfg.model.threshold_change.vv_db,
            vh_threshold_db=cfg.model.threshold_change.vh_db,
        )
        hist_probs.append(hist_p)

    if hist_probs:
        freq_agua_hist = xr.concat(hist_probs, dim="hist_window").mean(
            dim="hist_window", skipna=True
        )
    else:
        freq_agua_hist = xr.full_like(baseline_template, np.nan, dtype="float32")
    freq_agua_hist = freq_agua_hist.astype("float32")
    freq_agua_hist.name = "freq_agua_hist"
    freq_agua_hist = freq_agua_hist.rio.write_crs(baseline_template.rio.crs)

    freq_path = paths["intermediate"] / "freq_agua_hist.tif"
    save_geotiff(freq_agua_hist, freq_path)

    dist_to_perm = compute_distance_to_permwater(
        freq_agua_hist=freq_agua_hist,
        threshold=0.6,
        resolution_m=float(cfg.resolution_m),
    )

    preprocess_manifest: dict[str, Any] = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": cfg.run_name,
        "freq_agua_hist_path": str(freq_path),
        "targets": [],
    }

    for target_record in ingest_manifest["targets"]:
        key = target_record["key"]
        s1_path = target_record["s1"]
        s2_path = target_record.get("s2", "")
        LOGGER.info("Preprocess target window: %s", key)

        s1_ds = open_stack(s1_path, fallback_crs=cfg.crs_salida)
        s1_feat = compute_s1_target_features(s1_ds, baseline_feat)
        template = s1_feat["vv_db"]

        layers: list[xr.Dataset] = [s1_feat]

        freq_match = freq_agua_hist.rio.reproject_match(template)
        layers.append(freq_match.to_dataset(name="freq_agua_hist"))

        dist_match = dist_to_perm.rio.reproject_match(template)
        layers.append(dist_match.to_dataset(name="dist_to_permwater"))

        if dem_features is not None:
            dem_aligned = align_dataset_to_template(dem_features, template)
            layers.append(dem_aligned)

        if s2_path:
            s2_ds = open_stack(s2_path, fallback_crs=cfg.crs_salida)
            s2_feat = compute_s2_features(s2_ds)
            if s2_feat is not None:
                s2_aligned = align_dataset_to_template(s2_feat, template)
                layers.append(s2_aligned)

        feat_ds = xr.merge(layers, compat="override")
        p_water_initial = threshold_water_probability(
            feat_ds,
            vv_threshold_db=cfg.model.threshold_change.vv_db,
            vh_threshold_db=cfg.model.threshold_change.vh_db,
        )

        # external_labels.water es agua PERMANENTE (WorldCover): sirve como referencia
        # para las pseudo-etiquetas. external_labels.flood es el ground truth del evento
        # y solo se guarda aparte (flood_gt / water_gt) para evaluación.
        perm_water_ref = load_external_label(cfg.external_labels.water, template)
        ext_flood = load_external_label(cfg.external_labels.flood, template)

        slope_da = feat_ds["slope"] if "slope" in feat_ds else None
        dist_da = feat_ds["dist_to_permwater"] if "dist_to_permwater" in feat_ds else None

        water_labels = generate_water_pseudolabels(
            p_water_initial=p_water_initial,
            freq_agua_hist=feat_ds["freq_agua_hist"],
            slope=slope_da,
            permanent_ref=perm_water_ref,
        )
        flood_labels = generate_flood_pseudolabels(
            p_water_initial=p_water_initial,
            freq_agua_hist=feat_ds["freq_agua_hist"],
            slope=slope_da,
            dist_to_permwater=dist_da,
            permanent_ref=perm_water_ref,
        )

        label_vars = {
            "water_label": water_labels,
            "flood_label": flood_labels,
            "p_water_initial": p_water_initial,
        }
        if perm_water_ref is not None:
            label_vars["perm_water_ref"] = perm_water_ref.rename("perm_water_ref")
        label_vars.update(build_ground_truth(ext_flood, perm_water_ref))
        labels_ds = xr.Dataset(label_vars)

        feat_path = paths["intermediate"] / f"features_{key}.nc"
        label_path = paths["intermediate"] / f"labels_{key}.nc"
        feat_ds.to_netcdf(feat_path)
        labels_ds.to_netcdf(label_path)

        preprocess_manifest["targets"].append(
            {
                "key": key,
                "window": target_record["window"],
                "features_path": str(feat_path),
                "labels_path": str(label_path),
            }
        )

    preprocess_manifest_path = paths["metadata"] / "manifest_preprocess.json"
    write_json(preprocess_manifest_path, preprocess_manifest)
    LOGGER.info("Preprocess manifest written to %s", preprocess_manifest_path)
    return preprocess_manifest
