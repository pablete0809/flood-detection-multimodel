from __future__ import annotations

"""Pixel-aware metrics for the model comparison framework.

All functions accept only the pixels with known labels (label != -1).
Callers are responsible for filtering to valid pixels before calling these.
"""

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    jaccard_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def _safe_metric(func, *args, default=float("nan"), **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:  # noqa: BLE001
        return default


def compute_all_metrics(
    y_true: np.ndarray,  # int, values in {0, 1}
    y_prob: np.ndarray,  # float32, values in [0, 1]
) -> dict[str, float]:
    """Compute AUC-ROC, F1, PR-AUC, IoU, Brier, ECE for a single model/window."""
    y_hat = (y_prob >= 0.5).astype(np.uint8)
    return {
        "auc_roc": _safe_metric(roc_auc_score, y_true, y_prob),
        "f1": _safe_metric(f1_score, y_true, y_hat, zero_division=0.0),
        "pr_auc": _safe_metric(average_precision_score, y_true, y_prob),
        "iou": _safe_metric(jaccard_score, y_true, y_hat, zero_division=0.0),
        "brier": _safe_metric(brier_score_loss, y_true, y_prob),
        "ece": _ece(y_true, y_prob),
    }


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        ece += abs(y_prob[mask].mean() - y_true[mask].mean()) * (mask.sum() / n)
    return float(ece)


def get_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (fpr, tpr, auc_roc)."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))
        return fpr, tpr, auc
    except Exception:  # noqa: BLE001
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")


def get_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (precision, recall, pr_auc)."""
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auc = float(average_precision_score(y_true, y_prob))
        return prec, rec, auc
    except Exception:  # noqa: BLE001
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), float("nan")


def get_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (prob_true, prob_pred) for reliability diagram."""
    try:
        return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    except Exception:  # noqa: BLE001
        return np.linspace(0, 1, n_bins), np.linspace(0, 1, n_bins)


def aggregate_metrics(
    per_window: list[dict[str, Any]]
) -> dict[str, float]:
    """Average per-window metrics across windows, ignoring NaNs."""
    keys = {k for entry in per_window for k in entry}
    result = {}
    for k in keys:
        vals = [entry[k] for entry in per_window if k in entry and not np.isnan(entry[k])]
        result[k] = float(np.mean(vals)) if vals else float("nan")
    return result
