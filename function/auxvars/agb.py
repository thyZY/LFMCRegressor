from __future__ import annotations
import os
import numpy as np
from .utils_h5 import read_h5_dataset, ensure_global_shape

def agb_file_for_year(agb_dir: str, year: int) -> str:
    # filename: GlobBiomassYYYY.h5
    return os.path.join(agb_dir, f"GlobBiomass{year}.h5")

def load_agb_static_2015(agb_dir: str, var_name: str = "AGB") -> np.ndarray:
    """
    As per your request: use 2015 AGB as static variable.
    """
    fp = agb_file_for_year(agb_dir, 2015)
    arr = read_h5_dataset(fp, var_name)
    arr = ensure_global_shape(arr)
    return arr.astype(np.float32)
