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

from floodprob.compare_metrics import best_f1_threshold
from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import reference_labels, training_labels
from floodprob.splits import SPLIT_TEST, SPLIT_VAL, split_for_cfg
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


def _val_threshold(y_sel: np.ndarray, p: np.ndarray, split: np.ndarray) -> float:
    """F1-optimal threshold on VAL pixels using the training-type labels."""
    m = (split == SPLIT_VAL) & (y_sel >= 0) & np.isfinite(p)
    return best_f1_threshold(y_sel[m].astype(np.uint8), p[m])


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, y_hat: np.ndarray
) -> dict[str, float]:
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
    water_hat_all: list[np.ndarray] = []
    flood_true_all: list[np.ndarray] = []
    flood_prob_all: list[np.ndarray] = []
    flood_hat_all: list[np.ndarray] = []

    train_on = cfg.model.train_on
    per_target = []
    for target in pre["targets"]:
        key = target["key"]
        if key not in pred_by_key:
            LOGGER.warning("Prediction not found for target key=%s", key)
            continue
        p_record = pred_by_key[key]
        labels_ds = xr.open_dataset(target["labels_path"])
        water_ref, water_src = reference_labels(labels_ds, "water")
        flood_ref, flood_src = reference_labels(labels_ds, "flood")
        water_labels = water_ref.values.astype(np.int16)
        flood_labels = flood_ref.values.astype(np.int16)

        p_water = _read_single_band(p_record["p_agua"]).values.astype(np.float32)
        p_flood_gw = _read_single_band(p_record["p_inundacion_dado_agua"]).values.astype(
            np.float32
        )

        split = split_for_cfg(p_water.shape, cfg)
        test = split == SPLIT_TEST
        water_thr = _val_threshold(
            training_labels(labels_ds, "water", train_on).values, p_water, split
        )
        flood_thr = _val_threshold(
            training_labels(labels_ds, "flood", train_on).values, p_flood_gw, split
        )

        water_mask = (water_labels >= 0) & np.isfinite(p_water) & test
        flood_mask = (flood_labels >= 0) & np.isfinite(p_flood_gw) & test

        if water_mask.sum() > 0:
            y_w = water_labels[water_mask].astype(np.uint8)
            p_w = np.clip(p_water[water_mask], 0.0, 1.0)
            h_w = (p_w >= water_thr).astype(np.uint8)
            water_true_all.append(y_w)
            water_prob_all.append(p_w)
            water_hat_all.append(h_w)
            water_metrics = _classification_metrics(y_w, p_w, h_w)
        else:
            water_metrics = {}

        if flood_mask.sum() > 0:
            y_f = flood_labels[flood_mask].astype(np.uint8)
            p_f = np.clip(p_flood_gw[flood_mask], 0.0, 1.0)
            h_f = (p_f >= flood_thr).astype(np.uint8)
            flood_true_all.append(y_f)
            flood_prob_all.append(p_f)
            flood_hat_all.append(h_f)
            flood_metrics = _classification_metrics(y_f, p_f, h_f)
        else:
            flood_metrics = {}

        per_target.append(
            {
                "key": key,
                "window": target["window"],
                "water_metrics": water_metrics,
                "flood_given_water_metrics": flood_metrics,
                "water_threshold": water_thr,
                "flood_threshold": flood_thr,
                "water_label_source": water_src,
                "flood_label_source": flood_src,
                "n_water_eval": int(water_mask.sum()),
                "n_flood_eval": int(flood_mask.sum()),
            }
        )

    overall = {}
    if water_true_all:
        yw = np.concatenate(water_true_all)
        pw = np.concatenate(water_prob_all)
        overall["water"] = _classification_metrics(yw, pw, np.concatenate(water_hat_all))
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
        overall["flood_given_water"] = _classification_metrics(
            yf, pf, np.concatenate(flood_hat_all)
        )
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
        "train_on": train_on,
        "spatial_holdout": {
            "strategy": "random_spatial_blocks",
            "tile_size_px": cfg.model.tile_size,
            "n_folds": cfg.model.n_folds,
            "test_fold": cfg.model.test_fold,
            "val_fold": cfg.model.val_fold,
            "seed": cfg.seed,
        },
        "overall": overall,
        "per_target": per_target,
    }
    report_path = paths["evaluation"] / "metrics.json"
    write_json(report_path, report)
    LOGGER.info("Evaluation report written to %s", report_path)
    return report
