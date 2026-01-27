from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
from .utils_h5 import read_h5_dataset, ensure_global_shape

IGBP_CLASSES: List[str] = [
    "ENF","EBF","DNF","DBF","MF","CSH","OSH","WSA","SAV","GRA","WET",
    "CRO","URB","CVM","SNO","BAR","WAT"
]

def igbp_file_for_year(igbp_dir: str, year: int) -> str:
    # filename: YYYY001.h5
    return os.path.join(igbp_dir, f"{year}001.h5")

def load_igbp_year(igbp_dir: str, year: int, classes: List[str] = IGBP_CLASSES) -> Dict[str, np.ndarray]:
    """
    Load IGBP class fractions (0-100 or 0-1 depending on your preprocessing).
    Returns dict[class_name] -> (1800,3600) float32.
    """
    fp = igbp_file_for_year(igbp_dir, year)
    out: Dict[str, np.ndarray] = {}
    for c in classes:
        arr = read_h5_dataset(fp, c)
        arr = ensure_global_shape(arr)
        out[c] = arr.astype(np.float32)
    return out

def find_available_igbp_years(igbp_dir: str) -> List[int]:
    years = []
    if not os.path.isdir(igbp_dir):
        return years
    for name in os.listdir(igbp_dir):
        if name.endswith(".h5") and len(name) >= 7:
            # "YYYY001.h5"
            try:
                y = int(name[:4])
                doy = name[4:7]
                if doy == "001":
                    years.append(y)
            except:
                pass
    return sorted(set(years))

def choose_igbp_year(target_year: int, available_years: List[int]) -> int:
    """
    Choose best IGBP year for a target year.
    Strategy: use the closest year <= target_year; if none, use the minimum available.
    """
    if not available_years:
        raise ValueError("No available IGBP years found.")
    le = [y for y in available_years if y <= target_year]
    return max(le) if le else min(available_years)
