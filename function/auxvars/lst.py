from __future__ import annotations
import os
import numpy as np
from datetime import datetime
from .utils_h5 import read_h5_dataset, ensure_global_shape

def lst_file_for_date(lst_dir: str, date: datetime) -> str:
    return os.path.join(lst_dir, date.strftime("%Y%m%d") + ".h5")

def load_lst_daily(lst_dir: str, date: datetime, var_name: str = "LST_Day") -> np.ndarray:
    """
    LST in Kelvin (as you stated).
    """
    fp = lst_file_for_date(lst_dir, date)
    arr = read_h5_dataset(fp, var_name)
    arr = ensure_global_shape(arr)
    return arr.astype(np.float32)
