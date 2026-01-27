from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd


def read_lfmc_xlsx(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Read Globe-LFMC xlsx (or similar) into a raw DataFrame.

    Required cfg keys:
      cfg["raw_path"]
      cfg.get("sheet_name", "LFMC Data")
    """
    path = cfg["raw_path"]
    sheet = cfg.get("sheet_name", "LFMC Data")

    # engine=openpyxl is the safe default for .xlsx
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    return df


def read_lfmc_raw(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Generic entry for reading raw LFMC.
    Currently supports xlsx via read_lfmc_xlsx.
    """
    raw_path = str(cfg.get("raw_path", "")).lower()
    if raw_path.endswith(".xlsx"):
        return read_lfmc_xlsx(cfg)

    raise ValueError(f"Unsupported LFMC file type: {cfg.get('raw_path')}")
