from __future__ import annotations

import numpy as np
from datetime import datetime

from .utils_h5 import read_h5_dataset, ensure_global_shape
from .utils_time_index import build_file_path_by_date


def load_fvc_from_file(file_path: str, var_name: str = "FVC") -> np.ndarray:
    arr = read_h5_dataset(file_path, var_name)
    arr = ensure_global_shape(arr)
    return arr.astype(np.float32)


def load_fvc_by_index(fvc_dir: str, indexed_date: datetime, var_name: str = "FVC") -> np.ndarray:
    """
    Load FVC using an already-chosen index date (e.g., 8-day window start).
    """
    fp = build_file_path_by_date(fvc_dir, indexed_date, suffix=".h5")
    return load_fvc_from_file(fp, var_name=var_name)
