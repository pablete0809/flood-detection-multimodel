from __future__ import annotations

"""Partición espacial por bloques, compartida por entrenamiento, evaluación y figuras.

La imagen se divide en tiles de tile_size × tile_size píxeles. Cada tile se asigna
aleatoriamente (con semilla) a uno de n_folds folds equilibrados; un fold es test,
otro validación y el resto entrenamiento. Los tiles están alineados con los patches
de las CNN (mismo origen y tamaño), así que ningún patch mezcla train y test.
"""

import numpy as np
import xarray as xr

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2


def spatial_split(
    shape: tuple[int, int],
    tile_size: int,
    seed: int,
    n_folds: int = 5,
    test_fold: int = 0,
    val_fold: int = 1,
) -> np.ndarray:
    """Devuelve un array (H, W) int8 con SPLIT_TRAIN / SPLIT_VAL / SPLIT_TEST."""
    if n_folds < 3:
        raise ValueError("n_folds must be >= 3 (train, val and test folds)")
    if test_fold == val_fold:
        raise ValueError("test_fold and val_fold must differ")

    H, W = shape
    n_ty = -(-H // tile_size)
    n_tx = -(-W // tile_size)
    rng = np.random.default_rng(seed)
    fold_of_tile = rng.permutation(n_ty * n_tx) % n_folds

    split_of_tile = np.full(n_ty * n_tx, SPLIT_TRAIN, dtype=np.int8)
    split_of_tile[fold_of_tile == test_fold] = SPLIT_TEST
    split_of_tile[fold_of_tile == val_fold] = SPLIT_VAL

    tile_id = (np.arange(H)[:, None] // tile_size) * n_tx + (np.arange(W)[None, :] // tile_size)
    return split_of_tile[tile_id]


def split_for_cfg(shape: tuple[int, int], cfg) -> np.ndarray:
    m = cfg.model
    return spatial_split(shape, m.tile_size, cfg.seed, m.n_folds, m.test_fold, m.val_fold)


def mask_to_train(labels: xr.DataArray, split: np.ndarray) -> xr.DataArray:
    """Copy of `labels` with every non-TRAIN pixel set to -1 (ignored by all trainers)."""
    return labels.copy(data=np.where(split == SPLIT_TRAIN, labels.values, -1).astype("int8"))
