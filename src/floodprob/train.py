from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import training_labels
from floodprob.models import train_flood_given_water, train_water
from floodprob.splits import mask_to_train, split_for_cfg
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


def run_train(cfg: AppConfig) -> dict:
    paths = build_run_dirs(cfg.outputs.out_dir, cfg.run_name)
    preprocess_manifest = _load_preprocess_manifest(cfg)

    water_train_sets = []
    flood_train_sets = []
    for target in preprocess_manifest["targets"]:
        feat_path = Path(target["features_path"])
        label_path = Path(target["labels_path"])
        feat_ds = xr.open_dataset(feat_path)
        labels_ds = xr.open_dataset(label_path)
        split = split_for_cfg(labels_ds["flood_label"].shape, cfg)

        water_y = training_labels(labels_ds, "water", cfg.model.train_on)
        flood_y = training_labels(labels_ds, "flood", cfg.model.train_on)
        water_train_sets.append((feat_ds, mask_to_train(water_y, split)))
        flood_train_sets.append(
            (feat_ds, mask_to_train(flood_y, split), labels_ds["p_water_initial"])
        )

    water_bundle = train_water.train_water_model_bundle(water_train_sets, cfg)
    flood_bundle = train_flood_given_water.train_flood_model_bundle(flood_train_sets, cfg)

    water_model_path = paths["models"] / "water_model.joblib"
    flood_model_path = paths["models"] / "flood_given_water_model.joblib"
    train_water.save_model_bundle(water_bundle, str(water_model_path))
    train_flood_given_water.save_model_bundle(flood_bundle, str(flood_model_path))

    report = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": cfg.run_name,
        "train_on": cfg.model.train_on,
        "water_model_path": str(water_model_path),
        "flood_model_path": str(flood_model_path),
        "water_model_type": water_bundle["model_type"],
        "flood_model_type": flood_bundle["model_type"],
        "water_features": water_bundle["feature_names"],
        "flood_features": flood_bundle["feature_names"],
    }
    report_path = paths["metadata"] / "manifest_train.json"
    write_json(report_path, report)
    LOGGER.info("Train manifest written to %s", report_path)
    return report
