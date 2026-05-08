from __future__ import annotations

"""Lightweight PyTorch U-Net for the *flood* head.

Architecture: 3-level encoder–decoder with skip connections.
  Channels per level: [C_in → 16 → 32 → 64] (encoder), mirrored decoder.
  Input  : patches of shape (B, C_features, tile_size, tile_size)
  Output : (B, 1, tile_size, tile_size)  — single-channel logit map, sigmoid → probability

CPU fallback: torch.device("cuda" if cuda.is_available() else "cpu")
Uncertainty : MC-Dropout (run forward N times with model.train() → mean + std)
Class imbalance: weighted BCEWithLogitsLoss(pos_weight = n_neg / n_pos)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

from floodprob.config import AppConfig
from floodprob.labels.pseudolabels import flatten_samples  # noqa: F401 (used indirectly)
from floodprob.models.train_flood_given_water import FLOOD_FEATURE_CANDIDATES

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def _get_unet():
    """Import torch and define the U-Net class. Returns (UNet class, torch module)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the UNet model. "
            "Install it with: pip install -e '.[torch]'"
        ) from exc

    class DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class UNet(nn.Module):
        def __init__(self, in_channels: int, dropout: float = 0.2):
            super().__init__()
            self.enc1 = DoubleConv(in_channels, 16, dropout)
            self.enc2 = DoubleConv(16, 32, dropout)
            self.enc3 = DoubleConv(32, 64, dropout)
            self.pool = nn.MaxPool2d(2)

            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec2 = DoubleConv(64, 32, dropout)  # 32 + 32 skip
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec1 = DoubleConv(32, 16, dropout)  # 16 + 16 skip
            self.out = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    return UNet, torch


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

def _extract_patches(
    arr: np.ndarray,  # (C, H, W) float32
    tile: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Extract non-overlapping patches.  Returns (N, C, tile, tile) and original (H, W)."""
    C, H, W = arr.shape
    # Pad to multiple of tile
    pad_h = (tile - H % tile) % tile
    pad_w = (tile - W % tile) % tile
    arr_padded = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = arr_padded.shape[1], arr_padded.shape[2]
    patches = arr_padded.reshape(C, Hp // tile, tile, Wp // tile, tile)
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(-1, C, tile, tile)
    return patches.astype(np.float32), (H, W)


def _reconstruct(
    patches: np.ndarray,  # (N, tile, tile) float32
    orig_shape: tuple[int, int],
    tile: int,
) -> np.ndarray:
    H, W = orig_shape
    pad_h = (tile - H % tile) % tile
    pad_w = (tile - W % tile) % tile
    Hp, Wp = H + pad_h, W + pad_w
    n_h, n_w = Hp // tile, Wp // tile
    grid = patches.reshape(n_h, n_w, tile, tile)
    full = grid.transpose(0, 2, 1, 3).reshape(Hp, Wp)
    return full[:H, :W]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_flood_unet_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]],
    cfg: AppConfig,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    UNet, torch = _get_unet()
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    feature_names = [
        f for f in FLOOD_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _, _ in train_sets)
    ]
    required = {"delta_vv", "delta_vh"}
    if not required.issubset(set(feature_names)):
        raise ValueError(f"UNet flood model requires features {sorted(required)}")

    unet_cfg = cfg.model.compare.unet
    tile = cfg.model.tile_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("UNet training on device: %s", device)

    # ---- Collect patches ----
    all_patches_X: list[np.ndarray] = []
    all_patches_y: list[np.ndarray] = []

    for ds, flood_labels, _p_water in train_sets:
        # Stack features → (C, H, W)
        feat_stack = np.stack([ds[f].values.astype(np.float32) for f in feature_names], axis=0)
        label_arr = flood_labels.values.astype(np.float32)  # (H, W)

        patches_X, _orig_shape = _extract_patches(feat_stack, tile)
        label_exp = label_arr[np.newaxis, :, :]  # (1, H, W)
        patches_y, _ = _extract_patches(label_exp, tile)  # (N, 1, tile, tile)

        # Keep only patches that have at least some labelled pixels
        has_label = (patches_y[:, 0, :, :] >= 0).any(axis=(1, 2))
        all_patches_X.append(patches_X[has_label])
        all_patches_y.append(patches_y[has_label, 0, :, :])

    if not all_patches_X:
        raise ValueError("No UNet training patches collected")

    X_all = np.vstack(all_patches_X)   # (N, C, tile, tile)
    X_all = np.nan_to_num(X_all, nan=0.0)  # NaN features (e.g. cloudy NDWI) → 0
    y_all = np.stack([p for arr in all_patches_y for p in arr])  # (N, tile, tile)
    LOGGER.info("UNet patches: %s  features: %s  tile: %s", len(X_all), len(feature_names), tile)

    # ---- Class weight ----
    valid_mask = y_all >= 0
    n_pos = float((y_all[valid_mask] == 1).sum())
    n_neg = float((y_all[valid_mask] == 0).sum())
    pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)

    # Convert; mask ignore pixels (-1) as 0 weight
    y_tensor = torch.from_numpy(np.where(y_all >= 0, y_all, 0.0).astype(np.float32))
    w_tensor = torch.from_numpy(valid_mask.astype(np.float32))

    dataset = TensorDataset(
        torch.from_numpy(X_all),
        y_tensor,
        w_tensor,
    )
    loader = DataLoader(dataset, batch_size=unet_cfg.batch_size, shuffle=True, drop_last=False)

    model = UNet(in_channels=len(feature_names), dropout=unet_cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=unet_cfg.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    model.train()
    for epoch in range(unet_cfg.epochs):
        epoch_loss = 0.0
        for Xb, yb, wb in loader:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            logits = model(Xb).squeeze(1)  # (B, tile, tile)
            loss = (criterion(logits, yb) * wb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            LOGGER.info("UNet epoch %d/%d  loss=%.4f", epoch + 1, unet_cfg.epochs, epoch_loss / len(loader))

    # ---- Save model weights ----
    if bundle_dir is None:
        bundle_dir = Path(".")
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / "unet_flood.pt"
    torch.save(model.state_dict(), weights_path)
    LOGGER.info("UNet weights saved to %s", weights_path)

    return {
        "model_type": "unet",
        "feature_names": feature_names,
        "models": [str(weights_path)],  # path to .pt file
        "unet_config": {
            "in_channels": len(feature_names),
            "dropout": unet_cfg.dropout,
            "mc_samples": unet_cfg.mc_samples,
            "tile_size": tile,
        },
        "feature_importances": None,
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_flood_unet(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray]:
    """MC-Dropout inference.  Returns (mean, std) DataArrays."""
    UNet, torch = _get_unet()

    unet_meta = bundle["unet_config"]
    tile = unet_meta["tile_size"]
    mc_n = unet_meta["mc_samples"]
    dropout = unet_meta["dropout"]
    in_channels = unet_meta["in_channels"]
    weights_path = Path(bundle["models"][0])
    feature_names = bundle["feature_names"]

    missing = [n for n in feature_names if n not in feature_ds]
    if missing:
        raise ValueError(f"Missing features for UNet: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=in_channels, dropout=dropout).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    feat_stack = np.stack([feature_ds[f].values.astype(np.float32) for f in feature_names], axis=0)
    feat_stack = np.nan_to_num(feat_stack, nan=0.0)
    patches_X, orig_shape = _extract_patches(feat_stack, tile)

    template_name = feature_names[0]
    template = feature_ds[template_name]

    # MC-Dropout: keep dropout active during inference
    model.train()
    mc_results: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(mc_n):
            run_probs: list[np.ndarray] = []
            bs = 16
            for i in range(0, len(patches_X), bs):
                Xb = torch.from_numpy(patches_X[i : i + bs]).to(device)
                logits = model(Xb).squeeze(1).cpu().numpy()  # (b, tile, tile)
                probs = 1.0 / (1.0 + np.exp(-logits))
                run_probs.append(probs)
            mc_results.append(np.concatenate(run_probs, axis=0))  # (N, tile, tile)

    mc_stack = np.stack(mc_results, axis=0)  # (mc_n, N, tile, tile)
    mean_patches = mc_stack.mean(axis=0)     # (N, tile, tile)
    std_patches  = mc_stack.std(axis=0)

    mean_map = _reconstruct(mean_patches, orig_shape, tile).astype(np.float32)
    std_map  = _reconstruct(std_patches,  orig_shape, tile).astype(np.float32)

    crs = template.rio.crs

    def _make_da(arr, name):
        da = xr.DataArray(arr, coords=template.coords, dims=template.dims, name=name).astype("float32")
        if crs:
            da = da.rio.write_crs(crs)
        return da

    return _make_da(mean_map, "p_unet"), _make_da(std_map, "p_unet_std")
