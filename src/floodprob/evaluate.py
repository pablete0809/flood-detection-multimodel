from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, jaccard_score, roc_auc_score

from floodprob.config import AppConfig
from floodprob.utils.io import build_run_dirs, read_json, write_json

LOGGER = logging.getLogger(__name__)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += abs(conf - acc) * (mask.sum() / n)
    return float(ece)


def _reliability_plot(
    y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _tile_holdout_mask(shape: tuple[int, int], tile_size: int) -> np.ndarray:
    y_idx = np.arange(shape[0])[:, None]
    x_idx = np.arange(shape[1])[None, :]
    tile_y = y_idx // tile_size
    tile_x = x_idx // tile_size
    tile_id = tile_y * 10000 + tile_x
    return (tile_id % 5) == 0


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_hat = (y_prob >= 0.5).astype(np.uint8)
    metrics = {
        "f1": float(f1_score(y_true, y_hat)),
        "iou": float(jaccard_score(y_true, y_hat)),
        "auc": _safe_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": _ece(y_true, y_prob),
    }
    return metrics


def _read_single_band(path: str | Path) -> xr.DataArray:
    return rioxarray.open_rasterio(path, masked=True).squeeze(drop=True)


def run_evaluate(cfg: AppConfig) -> dict:
    paths = build_run_dirs(cfg.outputs.out_dir, cfg.run_name)
    preprocess_manifest_path = paths["metadata"] / "manifest_preprocess.json"
    predict_manifest_path = paths["metadata"] / "manifest_predict.json"
    if not preprocess_manifest_path.exists() or not predict_manifest_path.exists():
        raise FileNotFoundError(
            "Missing preprocess/predict manifests. Run preprocess and predict first."
        )

    pre = read_json(preprocess_manifest_path)
    pred = read_json(predict_manifest_path)
    pred_by_key = {r["key"]: r for r in pred["predictions"]}

    water_true_all: list[np.ndarray] = []
    water_prob_all: list[np.ndarray] = []
    flood_true_all: list[np.ndarray] = []
    flood_prob_all: list[np.ndarray] = []

    per_target = []
    for target in pre["targets"]:
        key = target["key"]
        if key not in pred_by_key:
            LOGGER.warning("Prediction not found for target key=%s", key)
            continue
        p_record = pred_by_key[key]
        labels_ds = xr.open_dataset(target["labels_path"])
        water_labels = labels_ds["water_label"].values.astype(np.int16)
        flood_labels = labels_ds["flood_label"].values.astype(np.int16)

        p_water = _read_single_band(p_record["p_agua"]).values.astype(np.float32)
        p_flood_gw = _read_single_band(p_record["p_inundacion_dado_agua"]).values.astype(
            np.float32
        )

        holdout = _tile_holdout_mask(p_water.shape, tile_size=cfg.model.tile_size)

        water_mask = (water_labels >= 0) & np.isfinite(p_water) & holdout
        flood_mask = (flood_labels >= 0) & np.isfinite(p_flood_gw) & holdout

        if water_mask.sum() > 0:
            y_w = water_labels[water_mask].astype(np.uint8)
            p_w = np.clip(p_water[water_mask], 0.0, 1.0)
            water_true_all.append(y_w)
            water_prob_all.append(p_w)
            water_metrics = _classification_metrics(y_w, p_w)
        else:
            water_metrics = {}

        if flood_mask.sum() > 0:
            y_f = flood_labels[flood_mask].astype(np.uint8)
            p_f = np.clip(p_flood_gw[flood_mask], 0.0, 1.0)
            flood_true_all.append(y_f)
            flood_prob_all.append(p_f)
            flood_metrics = _classification_metrics(y_f, p_f)
        else:
            flood_metrics = {}

        per_target.append(
            {
                "key": key,
                "window": target["window"],
                "water_metrics": water_metrics,
                "flood_given_water_metrics": flood_metrics,
                "n_water_eval": int(water_mask.sum()),
                "n_flood_eval": int(flood_mask.sum()),
            }
        )

    overall = {}
    if water_true_all:
        yw = np.concatenate(water_true_all)
        pw = np.concatenate(water_prob_all)
        overall["water"] = _classification_metrics(yw, pw)
        if len(np.unique(yw)) > 1:
            _reliability_plot(
                yw,
                pw,
                title="Reliability - Water",
                out_path=paths["evaluation"] / "reliability_water.png",
            )
    else:
        overall["water"] = {}

    if flood_true_all:
        yf = np.concatenate(flood_true_all)
        pf = np.concatenate(flood_prob_all)
        overall["flood_given_water"] = _classification_metrics(yf, pf)
        if len(np.unique(yf)) > 1:
            _reliability_plot(
                yf,
                pf,
                title="Reliability - Flood Given Water",
                out_path=paths["evaluation"] / "reliability_flood.png",
            )
    else:
        overall["flood_given_water"] = {}

    report = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_name": cfg.run_name,
        "spatial_holdout": {
            "strategy": "tile_modulo_5",
            "tile_size_px": cfg.model.tile_size,
        },
        "overall": overall,
        "per_target": per_target,
    }
    report_path = paths["evaluation"] / "metrics.json"
    write_json(report_path, report)
    LOGGER.info("Evaluation report written to %s", report_path)
    return report
