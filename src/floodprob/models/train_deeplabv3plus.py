from __future__ import annotations

"""DeepLabV3+ para el flood head, usando segmentation-models-pytorch.

DeepLabV3+ (Chen et al. 2018) combina ASPP (Atrous Spatial Pyramid Pooling)
con un decoder ligero y conexiones skip del encoder. A diferencia del U-Net
baseline, el encoder usa convoluciones dilatadas para capturar contexto
multi-escala sin reducir resolución.

Se entrena desde cero (encoder_weights=None) ya que los datos SAR no se
asemejan a imágenes RGB de ImageNet.

Inferencia: pase único con model.eval() (sin MC-Dropout).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from floodprob.config import AppConfig
from floodprob.models.train_flood_given_water import FLOOD_FEATURE_CANDIDATES
from floodprob.models.train_unet import _extract_patches, _reconstruct

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _get_deeplabv3plus_model(in_channels: int, encoder: str):
    """Devuelve (model, torch). Requiere PyTorch + segmentation-models-pytorch."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for DeepLabV3+. "
            "Install it with: pip install -e '.[torch]'"
        ) from exc
    try:
        import segmentation_models_pytorch as smp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "segmentation-models-pytorch is required for DeepLabV3+. "
            "Install it with: pip install -e '.[compare]'"
        ) from exc

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,      # train from scratch — SAR ≠ ImageNet
        in_channels=in_channels,
        classes=1,
        activation=None,           # raw logits; sigmoid applied at inference
    )
    return model, torch


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_flood_deeplabv3plus_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]],
    cfg: AppConfig,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    dlv3_cfg = cfg.model.compare.deeplabv3plus
    model, torch = _get_deeplabv3plus_model(
        in_channels=1,          # placeholder; will be overwritten below
        encoder=dlv3_cfg.encoder,
    )

    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    feature_names = [
        f for f in FLOOD_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _, _ in train_sets)
    ]
    required = {"delta_vv", "delta_vh"}
    if not required.issubset(set(feature_names)):
        raise ValueError(f"DeepLabV3+ requires features {sorted(required)}")

    # Rebuild model with correct in_channels
    model, torch = _get_deeplabv3plus_model(len(feature_names), dlv3_cfg.encoder)

    tile   = cfg.model.tile_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("DeepLabV3+ training on device: %s  encoder: %s", device, dlv3_cfg.encoder)

    # ---- Collect patches ----
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for ds, flood_labels, _p_water in train_sets:
        feat_stack = np.stack([ds[f].values.astype(np.float32) for f in feature_names], axis=0)
        label_arr  = flood_labels.values.astype(np.float32)

        patches_X, _ = _extract_patches(feat_stack, tile)
        patches_y, _ = _extract_patches(label_arr[np.newaxis, :, :], tile)

        has_label = (patches_y[:, 0, :, :] >= 0).any(axis=(1, 2))
        all_X.append(patches_X[has_label])
        all_y.append(patches_y[has_label, 0, :, :])

    if not all_X:
        raise ValueError("No DeepLabV3+ training patches collected")

    X_all = np.vstack(all_X)
    X_all = np.nan_to_num(X_all, nan=0.0)
    y_all = np.concatenate(all_y, axis=0)
    LOGGER.info("DeepLabV3+ patches: %s  features: %s  tile: %s", len(X_all), len(feature_names), tile)

    # ---- Class weight ----
    valid_mask = y_all >= 0
    n_pos = float((y_all[valid_mask] == 1).sum())
    n_neg = float((y_all[valid_mask] == 0).sum())
    pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    LOGGER.info("DeepLabV3+ pos_weight=%.2f  (n_pos=%d  n_neg=%d)", pos_weight_val, int(n_pos), int(n_neg))

    y_tensor = torch.from_numpy(np.where(y_all >= 0, y_all, 0.0).astype(np.float32))
    w_tensor = torch.from_numpy(valid_mask.astype(np.float32))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_all), y_tensor, w_tensor),
        batch_size=dlv3_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=dlv3_cfg.lr)
    # Cosine LR annealing: suave para redes más grandes
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dlv3_cfg.epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    model.train()
    for epoch in range(dlv3_cfg.epochs):
        epoch_loss = 0.0
        for Xb, yb, wb in loader:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            logits = model(Xb).squeeze(1)   # smp output: (B, 1, H, W) → squeeze → (B, H, W)
            loss = (criterion(logits, yb) * wb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            LOGGER.info(
                "DeepLabV3+ epoch %d/%d  loss=%.4f  lr=%.2e",
                epoch + 1, dlv3_cfg.epochs, epoch_loss / len(loader),
                scheduler.get_last_lr()[0],
            )

    if bundle_dir is None:
        bundle_dir = Path(".")
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / "deeplabv3plus_flood.pt"
    torch.save(model.state_dict(), weights_path)
    LOGGER.info("DeepLabV3+ weights saved to %s", weights_path)

    return {
        "model_type": "deeplabv3plus",
        "feature_names": feature_names,
        "models": [str(weights_path)],
        "deeplabv3plus_config": {
            "in_channels": len(feature_names),
            "encoder": dlv3_cfg.encoder,
            "tile_size": tile,
        },
        "feature_importances": None,
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_flood_deeplabv3plus(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
) -> tuple[xr.DataArray, None]:
    """Pase único determinista. Devuelve (mean_da, None) para consistencia con la API."""
    meta          = bundle["deeplabv3plus_config"]
    tile          = meta["tile_size"]
    in_channels   = meta["in_channels"]
    encoder       = meta["encoder"]
    weights_path  = Path(bundle["models"][0])
    feature_names = bundle["feature_names"]

    missing = [n for n in feature_names if n not in feature_ds]
    if missing:
        raise ValueError(f"Missing features for DeepLabV3+: {missing}")

    model, torch = _get_deeplabv3plus_model(in_channels, encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    feat_stack = np.stack([feature_ds[f].values.astype(np.float32) for f in feature_names], axis=0)
    feat_stack = np.nan_to_num(feat_stack, nan=0.0)
    patches_X, orig_shape = _extract_patches(feat_stack, tile)

    template = feature_ds[feature_names[0]]
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        bs = 8  # batch size más pequeño para evitar OOM con resnet34
        for i in range(0, len(patches_X), bs):
            Xb = torch.from_numpy(patches_X[i : i + bs]).to(device)
            logits = model(Xb).squeeze(1).cpu().numpy()  # (b, tile, tile)
            all_probs.append(1.0 / (1.0 + np.exp(-logits)))

    prob_patches = np.concatenate(all_probs, axis=0)      # (N, tile, tile)
    mean_map = _reconstruct(prob_patches, orig_shape, tile).astype(np.float32)

    crs = template.rio.crs
    da = xr.DataArray(
        mean_map, coords=template.coords, dims=template.dims, name="p_deeplabv3plus"
    ).astype("float32")
    if crs:
        da = da.rio.write_crs(crs)

    return da, None
