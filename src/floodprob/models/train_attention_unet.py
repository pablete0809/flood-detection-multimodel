from __future__ import annotations

"""Attention U-Net (Oktay et al. 2018) for the flood head.

Architecture: mismo esquema encoder–decoder 3 niveles que el U-Net base,
pero con puertas de atención (additive attention gates) en las conexiones
skip del decoder. Las puertas suprimen features de encoder irrelevantes y
potencian las regiones de interés (áreas inundadas).

Reutiliza los mismos patch helpers de train_unet.py.
MC-Dropout igual que en el U-Net base.
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
# Architecture
# ---------------------------------------------------------------------------

def _get_attention_unet():
    """Import torch and define the Attention U-Net. Returns (AttentionUNet class, torch)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for AttentionUNet. "
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

    class AttentionGate(nn.Module):
        """Soft attention gate — additive formulation (Oktay et al. 2018, Fig. 2).

        g : señal de gating del decoder (mismo tamaño espacial que x tras upsample)
        x : skip connection del encoder
        Ambos deben tener las mismas dimensiones espaciales.
        """
        def __init__(self, F_g: int, F_l: int, F_int: int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
                nn.BatchNorm2d(F_int),
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
                nn.BatchNorm2d(F_int),
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid(),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g: "torch.Tensor", x: "torch.Tensor") -> "torch.Tensor":
            psi = self.psi(self.relu(self.W_g(g) + self.W_x(x)))  # (B,1,H,W)
            return x * psi  # broadcast over channels

    class AttentionUNet(nn.Module):
        """3-level U-Net with attention gates on all skip connections."""
        def __init__(self, in_channels: int, dropout: float = 0.2):
            super().__init__()
            # Encoder
            self.enc1 = DoubleConv(in_channels, 16, dropout)
            self.enc2 = DoubleConv(16, 32, dropout)
            self.enc3 = DoubleConv(32, 64, dropout)
            self.pool = nn.MaxPool2d(2)

            # Decoder level 2 (e3 → e2 scale)
            self.up2  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.att2 = AttentionGate(F_g=32, F_l=32, F_int=16)
            self.dec2 = DoubleConv(64, 32, dropout)   # 32 from up + 32 attended skip

            # Decoder level 1 (d2 → e1 scale)
            self.up1  = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
            self.att1 = AttentionGate(F_g=16, F_l=16, F_int=8)
            self.dec1 = DoubleConv(32, 16, dropout)   # 16 from up + 16 attended skip

            self.out  = nn.Conv2d(16, 1, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))

            g2 = self.up2(e3)
            d2 = self.dec2(torch.cat([g2, self.att2(g2, e2)], dim=1))

            g1 = self.up1(d2)
            d1 = self.dec1(torch.cat([g1, self.att1(g1, e1)], dim=1))

            return self.out(d1)

    return AttentionUNet, torch


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_flood_attention_unet_bundle(
    train_sets: list[tuple[xr.Dataset, xr.DataArray, xr.DataArray]],
    cfg: AppConfig,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    AttentionUNet, torch = _get_attention_unet()
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    feature_names = [
        f for f in FLOOD_FEATURE_CANDIDATES if any(f in ds.data_vars for ds, _, _ in train_sets)
    ]
    required = {"delta_vv", "delta_vh"}
    if not required.issubset(set(feature_names)):
        raise ValueError(f"AttentionUNet requires features {sorted(required)}")

    au_cfg = cfg.model.compare.attention_unet
    tile   = cfg.model.tile_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("AttentionUNet training on device: %s", device)

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
        raise ValueError("No AttentionUNet training patches collected")

    X_all = np.vstack(all_X)
    X_all = np.nan_to_num(X_all, nan=0.0)
    y_all = np.concatenate(all_y, axis=0)
    LOGGER.info("AttentionUNet patches: %s  features: %s  tile: %s", len(X_all), len(feature_names), tile)

    # ---- Class weight ----
    valid_mask = y_all >= 0
    n_pos = float((y_all[valid_mask] == 1).sum())
    n_neg = float((y_all[valid_mask] == 0).sum())
    pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    LOGGER.info("AttentionUNet pos_weight=%.2f  (n_pos=%d  n_neg=%d)", pos_weight_val, int(n_pos), int(n_neg))

    y_tensor = torch.from_numpy(np.where(y_all >= 0, y_all, 0.0).astype(np.float32))
    w_tensor = torch.from_numpy(valid_mask.astype(np.float32))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_all), y_tensor, w_tensor),
        batch_size=au_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = AttentionUNet(in_channels=len(feature_names), dropout=au_cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=au_cfg.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    model.train()
    for epoch in range(au_cfg.epochs):
        epoch_loss = 0.0
        for Xb, yb, wb in loader:
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            logits = model(Xb).squeeze(1)
            loss = (criterion(logits, yb) * wb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            LOGGER.info(
                "AttentionUNet epoch %d/%d  loss=%.4f",
                epoch + 1, au_cfg.epochs, epoch_loss / len(loader),
            )

    if bundle_dir is None:
        bundle_dir = Path(".")
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / "attention_unet_flood.pt"
    torch.save(model.state_dict(), weights_path)
    LOGGER.info("AttentionUNet weights saved to %s", weights_path)

    return {
        "model_type": "attention_unet",
        "feature_names": feature_names,
        "models": [str(weights_path)],
        "unet_config": {
            "in_channels": len(feature_names),
            "dropout": au_cfg.dropout,
            "mc_samples": au_cfg.mc_samples,
            "tile_size": tile,
        },
        "feature_importances": None,
    }


# ---------------------------------------------------------------------------
# Predict (MC-Dropout)
# ---------------------------------------------------------------------------

def predict_flood_attention_unet(
    feature_ds: xr.Dataset,
    bundle: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray]:
    """MC-Dropout inference. Returns (mean, std) DataArrays."""
    AttentionUNet, torch = _get_attention_unet()

    meta         = bundle["unet_config"]
    tile         = meta["tile_size"]
    mc_n         = meta["mc_samples"]
    dropout      = meta["dropout"]
    in_channels  = meta["in_channels"]
    weights_path = Path(bundle["models"][0])
    feature_names = bundle["feature_names"]

    missing = [n for n in feature_names if n not in feature_ds]
    if missing:
        raise ValueError(f"Missing features for AttentionUNet: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=in_channels, dropout=dropout).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    feat_stack = np.stack([feature_ds[f].values.astype(np.float32) for f in feature_names], axis=0)
    feat_stack = np.nan_to_num(feat_stack, nan=0.0)
    patches_X, orig_shape = _extract_patches(feat_stack, tile)

    template = feature_ds[feature_names[0]]

    n_passes = max(mc_n, 1)
    model.train()  # keep dropout active for MC sampling
    mc_results: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_passes):
            run_probs: list[np.ndarray] = []
            bs = 16
            for i in range(0, len(patches_X), bs):
                Xb = torch.from_numpy(patches_X[i : i + bs]).to(device)
                logits = model(Xb).squeeze(1).cpu().numpy()
                run_probs.append(1.0 / (1.0 + np.exp(-logits)))
            mc_results.append(np.concatenate(run_probs, axis=0))

    mc_stack = np.stack(mc_results, axis=0)          # (passes, N, tile, tile)
    mean_patches = mc_stack.mean(axis=0)
    std_patches  = mc_stack.std(axis=0)

    mean_map = _reconstruct(mean_patches, orig_shape, tile).astype(np.float32)
    std_map  = _reconstruct(std_patches,  orig_shape, tile).astype(np.float32)

    crs = template.rio.crs

    def _make_da(arr: np.ndarray, name: str) -> xr.DataArray:
        da = xr.DataArray(arr, coords=template.coords, dims=template.dims, name=name).astype("float32")
        if crs:
            da = da.rio.write_crs(crs)
        return da

    return _make_da(mean_map, "p_attention_unet"), _make_da(std_map, "p_attention_unet_std")
