from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import rioxarray  # noqa: F401
import xarray as xr

from floodprob.config import AppConfig
from floodprob.models import train_flood_given_water, train_water
from floodprob.utils.geospatial import save_geotiff
from floodprob.utils.io import build_run_dirs, read_json, write_json

LOGGER = logging.getLogger(__name__)


def _load_preprocess_manifest(cfg: AppConfig) -> dict:
    paths = build_run_dirs(cfg.outputs.out_dir, cfg.run_name)
    manifest_path = paths["metadata"] / "manifest_preprocess.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Preprocess manifest not found: {manifest_path}. Run `floodprob preprocess` first."
        )
    return read_json(manifest_path)


def run_predict(cfg: AppConfig) -> dict:
    paths = build_run_dirs(cfg.outputs.out_dir, cfg.run_name)
    preprocess_manifest = _load_preprocess_manifest(cfg)

    water_model_path = paths["models"] / "water_model.joblib"
    flood_model_path = paths["models"] / "flood_given_water_model.joblib"
    if not water_model_path.exists() or not flood_model_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `floodprob train --config ...` first."
        )

    water_bundle = train_water.load_model_bundle(str(water_model_path))
    flood_bundle = train_flood_given_water.load_model_bundle(str(flood_model_path))

    prediction_records: list[dict] = []
    for target in preprocess_manifest["targets"]:
        key = target["key"]
        feat_path = Path(target["features_path"])
        feat_ds = xr.open_dataset(feat_path)

        p_agua, p_agua_std = train_water.predict_water_probability(feat_ds, water_bundle)
        p_flood_gw, p_flood_gw_std = train_flood_given_water.predict_flood_probability(
            feat_ds,
            p_water=p_agua,
            bundle=flood_bundle,
        )
        p_flood = (p_agua * p_flood_gw).clip(0.0, 1.0).astype("float32")
        p_flood.name = "p_inundacion"
        p_flood = p_flood.rio.write_crs(p_agua.rio.crs or cfg.crs_salida)

        p_agua = p_agua.rio.write_crs(p_agua.rio.crs or cfg.crs_salida)
        p_flood_gw = p_flood_gw.rio.write_crs(p_flood_gw.rio.crs or cfg.crs_salida)

        p_agua_path = paths["predictions"] / f"p_agua_{key}.tif"
        p_flood_gw_path = paths["predictions"] / f"p_inundacion_dado_agua_{key}.tif"
        p_flood_path = paths["predictions"] / f"p_inundacion_{key}.tif"
        save_geotiff(p_agua, p_agua_path)
        save_geotiff(p_flood_gw, p_flood_gw_path)
        save_geotiff(p_flood, p_flood_path)

        rec = {
            "key": key,
            "window": target["window"],
            "p_agua": str(p_agua_path),
            "p_inundacion_dado_agua": str(p_flood_gw_path),
            "p_inundacion": str(p_flood_path),
        }
        if p_agua_std is not None:
            p_agua_std = p_agua_std.rio.write_crs(p_agua.rio.crs or cfg.crs_salida)
            p_agua_std_path = paths["predictions"] / f"p_agua_std_{key}.tif"
            save_geotiff(p_agua_std, p_agua_std_path)
            rec["p_agua_std"] = str(p_agua_std_path)
        if p_flood_gw_std is not None:
            p_flood_gw_std = p_flood_gw_std.rio.write_crs(
                p_flood_gw.rio.crs or cfg.crs_salida
            )
            p_flood_gw_std_path = (
                paths["predictions"] / f"p_inundacion_dado_agua_std_{key}.tif"
            )
            save_geotiff(p_flood_gw_std, p_flood_gw_std_path)
            rec["p_inundacion_dado_agua_std"] = str(p_flood_gw_std_path)
        prediction_records.append(rec)

    report = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": cfg.run_name,
        "predictions": prediction_records,
    }
    report_path = paths["metadata"] / "manifest_predict.json"
    write_json(report_path, report)
    LOGGER.info("Prediction manifest written to %s", report_path)
    return report
