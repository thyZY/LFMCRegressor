from __future__ import annotations

from typing import Any, Dict
import os
import json

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_parquet(df: pd.DataFrame, out_path: str) -> None:
    """
    Export DataFrame to Parquet with minimal, schema-safe dtype normalization.

    Why needed:
      Parquet/Arrow requires each column to have a single physical type.
      Pandas 'object' columns may contain mixed Python types (e.g., int+str),
      which will crash pyarrow. We fix only known mixed columns from Globe-LFMC.

    Rules (minimal change):
      - sampling_time_raw: mixed time/datetime/str -> store as string (lossless)
      - protocol: mixed int/str and may contain codes like '14b' -> store as string (lossless)
      - Slope (%): int/float/str -> store as float (numeric), unparseable -> NaN
    """
    ensure_dir(os.path.dirname(out_path))
    out = df.copy()

    # --- 1) sampling_time_raw: time/datetime/str -> string (lossless) ---
    if "sampling_time_raw" in out.columns:
        s = out["sampling_time_raw"]

        # Convert any datetime/time objects to ISO-like strings; keep existing strings.
        # NaN/None stay as <NA> after astype("string").
        def _time_to_str(x):
            if pd.isna(x):
                return None
            # datetime.datetime
            if hasattr(x, "isoformat") and not isinstance(x, str):
                # datetime.time also has isoformat()
                return x.isoformat()
            return str(x)

        out["sampling_time_raw"] = s.map(_time_to_str).astype("string")

    # --- 2) protocol: int/str -> string (lossless; supports '14b', '32c') ---
    if "protocol" in out.columns:
        s = out["protocol"]

        # normalize: strip, lower, drop trailing ".0" (Excel sometimes makes 16 -> 16.0)
        def _prot_norm(x):
            if pd.isna(x):
                return None
            t = str(x).strip().lower()
            if t.endswith(".0"):
                t = t[:-2]
            return t

        out["protocol"] = s.map(_prot_norm).astype("string")

    # --- 3) Slope (%): int/float/str -> float64 (numeric); unparseable -> NaN ---
    if "Slope (%)" in out.columns:
        # Keep it numeric for modeling/statistics
        out["Slope (%)"] = pd.to_numeric(out["Slope (%)"], errors="coerce").astype("float64")

    # --- 4) write parquet (pyarrow via pandas) ---
    out.to_parquet(out_path, index=False)



def export_report(report: Dict[str, Any], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_rows": int(len(df))}
    if "lfmc_pct" in df.columns:
        out["lfmc_min"] = float(df["lfmc_pct"].min(skipna=True))
        out["lfmc_max"] = float(df["lfmc_pct"].max(skipna=True))
        out["lfmc_mean"] = float(df["lfmc_pct"].mean(skipna=True))
    if "date" in df.columns:
        out["date_min"] = str(df["date"].min())
        out["date_max"] = str(df["date"].max())
    if "site_id" in df.columns:
        out["n_sites"] = int(df["site_id"].nunique(dropna=True))
    return out
