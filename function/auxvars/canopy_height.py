from __future__ import annotations
import numpy as np
from .utils_h5 import read_h5_dataset, ensure_global_shape

def load_canopy_height(h5_path: str, var_name: str = "Hveg") -> np.ndarray:
    """
    Static canopy height.
    Returns (1800,3600) float32.
    """
    arr = read_h5_dataset(h5_path, var_name)
    arr = ensure_global_shape(arr)
    return arr.astype(np.float32)
