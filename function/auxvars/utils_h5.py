from __future__ import annotations
import os
import h5py
import numpy as np


GLOBAL_SHAPE = (1800, 3600)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expect 2D array, got shape={arr.shape}")
    return arr


def ensure_global_shape(arr: np.ndarray, *, allow_transpose: bool = True) -> np.ndarray:
    """
    Ensure array shape is (1800,3600).
    If it's (3600,1800), transpose it.
    """
    arr = _ensure_2d(arr)

    if arr.shape == GLOBAL_SHAPE:
        return arr

    if allow_transpose and arr.shape == (GLOBAL_SHAPE[1], GLOBAL_SHAPE[0]):
        return arr.T

    raise ValueError(f"Unsupported global grid shape: {arr.shape}, expected {GLOBAL_SHAPE} or {(GLOBAL_SHAPE[1], GLOBAL_SHAPE[0])}")


def read_h5_dataset(file_path: str, dataset_name: str, *, squeeze: bool = True) -> np.ndarray:
    """
    Read a dataset from .h5 into numpy array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with h5py.File(file_path, "r") as f:
        if dataset_name not in f:
            # try case-insensitive match (common in some exports)
            keys = list(f.keys())
            lower_map = {k.lower(): k for k in keys}
            if dataset_name.lower() in lower_map:
                dataset_name = lower_map[dataset_name.lower()]
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}. Available: {keys[:30]}{'...' if len(keys)>30 else ''}")

        data = f[dataset_name][:]
        if squeeze:
            data = np.squeeze(data)
        return np.array(data)


def nan_to_num_safe(arr: np.ndarray, nan_value: float = np.nan) -> np.ndarray:
    """
    If you want to keep NaN, pass nan_value=np.nan (default).
    If you want fill NaN with 0, pass nan_value=0.0.
    """
    if np.isnan(nan_value):
        return arr
    return np.nan_to_num(arr, nan=nan_value)
