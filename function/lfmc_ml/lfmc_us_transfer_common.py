from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from lfmc_final_common import (
    DATE_COL,
    LAT_COL,
    LON_COL,
    RANDOM_SEED,
    SPLIT_BLOCK_DEG,
    VOD_COLS,
    Y_COL,
    build_final_s6_lite_frame,
    eval_regression,
    load_base_dataframe,
    save_result_tables,
)


COUNTRY_COL = "Country"
STATE_COL = "State/Region"

TRAIN_SPLIT = "train_us"
VAL_OVERLAP_SPLIT = "val_us_overlap"
VAL_NONOVERLAP_SPLIT = "val_us_nonoverlap"
EXTERNAL_SPLIT = "test_external"

SPLIT_ORDER = [
    TRAIN_SPLIT,
    VAL_OVERLAP_SPLIT,
    VAL_NONOVERLAP_SPLIT,
    EXTERNAL_SPLIT,
]


def normalize_country(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def normalize_region(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def is_us_mainland_or_alaska(df: pd.DataFrame) -> pd.Series:
    country = df[COUNTRY_COL].map(normalize_country)
    region = df[STATE_COL].map(normalize_region) if STATE_COL in df.columns else pd.Series("", index=df.index)

    us_mask = country.isin(
        {
            "usa",
            "u.s.a.",
            "us",
            "u.s.",
            "united states",
            "united states of america",
        }
    )
    hawaii_mask = region.str.contains("hawai", na=False)
    return us_mask & ~hawaii_mask


def add_block_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_block_lat"] = np.floor((out[LAT_COL] + 90.0) / SPLIT_BLOCK_DEG).astype(int)
    out["_block_lon"] = np.floor((out[LON_COL] + 180.0) / SPLIT_BLOCK_DEG).astype(int)
    out["block_id"] = out["_block_lat"].astype(str) + "_" + out["_block_lon"].astype(str)
    return out


def build_us_transfer_s6_lite_frame(
    data_path: Path | None = None,
) -> Tuple[pd.DataFrame, List[str], str]:
    base_df = load_base_dataframe(data_path)
    final_df, num_cols, cat_col = build_final_s6_lite_frame(base_df)

    keep_cols = [c for c in [COUNTRY_COL, STATE_COL] if c in base_df.columns]
    final_df = final_df.merge(
        base_df[[DATE_COL, LAT_COL, LON_COL] + keep_cols].drop_duplicates(),
        on=[DATE_COL, LAT_COL, LON_COL],
        how="left",
    )
    final_df = add_block_columns(final_df)

    dt = pd.to_datetime(final_df[DATE_COL], errors="coerce")
    year = dt.dt.year
    us_mask = is_us_mainland_or_alaska(final_df)

    train_mask = us_mask & year.between(2002, 2018, inclusive="both")
    future_us_mask = us_mask & year.between(2019, 2023, inclusive="both")
    external_mask = ~us_mask

    train_blocks = set(final_df.loc[train_mask, "block_id"].dropna().unique().tolist())
    overlap_mask = future_us_mask & final_df["block_id"].isin(train_blocks)
    nonoverlap_mask = future_us_mask & ~final_df["block_id"].isin(train_blocks)

    out = final_df.copy()
    out["eval_split"] = "drop"
    out.loc[train_mask, "eval_split"] = TRAIN_SPLIT
    out.loc[overlap_mask, "eval_split"] = VAL_OVERLAP_SPLIT
    out.loc[nonoverlap_mask, "eval_split"] = VAL_NONOVERLAP_SPLIT
    out.loc[external_mask, "eval_split"] = EXTERNAL_SPLIT

    out = out[out["eval_split"].isin(SPLIT_ORDER)].copy()
    out["split"] = out["eval_split"]
    return out, num_cols, cat_col


def split_frame_by_eval_group(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {split_name: df[df["eval_split"] == split_name].copy() for split_name in SPLIT_ORDER}


def print_us_transfer_dataset_summary(df: pd.DataFrame, num_cols: List[str], cat_col: str):
    print("=== US Transfer S6-lite dataset ===")
    print(f"rows: {len(df)}")
    print("numeric columns:", num_cols)
    print("categorical column:", cat_col)
    for split_name in SPLIT_ORDER:
        sub = df[df["eval_split"] == split_name]
        print(f"{split_name}: n={len(sub)}, blocks={sub['block_id'].nunique()}")


def print_us_transfer_metrics(model_name: str, repeat_idx: int, metrics_by_split: Dict[str, Dict[str, float]]):
    print(f"[repeat {repeat_idx}] {model_name} metrics")
    for split_name in SPLIT_ORDER:
        m = metrics_by_split[split_name]
        print(
            f"  {split_name:<18} "
            f"MAE={m['MAE']:.4f}  "
            f"RMSE={m['RMSE']:.4f}  "
            f"R={m['R']:.4f}"
        )


def collect_us_transfer_rows(
    rows: List[Dict],
    model_name: str,
    repeat_idx: int,
    split_frames: Dict[str, pd.DataFrame],
    preds_by_split: Dict[str, np.ndarray],
    scheme: str,
    scheme_desc: str,
):
    metrics_by_split = {}
    for split_name in SPLIT_ORDER:
        split_df = split_frames[split_name]
        metrics = eval_regression(split_df[Y_COL].to_numpy(), preds_by_split[split_name])
        metrics_by_split[split_name] = metrics
        rows.append(
            {
                "scheme": scheme,
                "scheme_desc": scheme_desc,
                "model": model_name,
                "repeat": repeat_idx,
                "split": split_name,
                "n": len(split_df),
                **metrics,
            }
        )
    return metrics_by_split


__all__ = [
    "COUNTRY_COL",
    "STATE_COL",
    "TRAIN_SPLIT",
    "VAL_OVERLAP_SPLIT",
    "VAL_NONOVERLAP_SPLIT",
    "EXTERNAL_SPLIT",
    "SPLIT_ORDER",
    "RANDOM_SEED",
    "build_us_transfer_s6_lite_frame",
    "split_frame_by_eval_group",
    "print_us_transfer_dataset_summary",
    "print_us_transfer_metrics",
    "collect_us_transfer_rows",
    "save_result_tables",
]
