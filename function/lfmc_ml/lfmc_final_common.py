from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as skm

# =======================
# Config
# =======================
DATA_PATH = Path(
    r"G:\data\Globe LFMC\country_statistics\globe_lfmc_filtered_step3_daily_0p1deg_with_predictors.csv"
)
OUT_DIR = Path(r"d:\Python\jupyter\jupyter\LFMCRegressor\figs\batch_runs_final")

DATE_COL = "Sampling date (YYYYMMDD)"
LAT_COL = "Latitude (WGS84, EPSG:4326)"
LON_COL = "Longitude (WGS84, EPSG:4326)"
Y_COL = "LFMC value (%)"

FT_COL = "FT"
IGBP_COL = "IGBP Land Cover"

VOD_COLS = [
    "VOD_Ku_Hpol_Asc",
    "VOD_Ku_Vpol_Asc",
    "VOD_X_Hpol_Asc",
    "VOD_X_Vpol_Asc",
    "VOD_C_Hpol_Asc",
    "VOD_C_Vpol_Asc",
]

# S6-lite core: keep LAI / Hveg / LST, drop SM
CORE_NUM_COLS_LITE = ["LAI", "Hveg", "LST"]
SEASON_COLS = ["doy_sin", "doy_cos", "lat_norm"]

LC_VEG_COLS = [
    "ENF", "EBF", "DNF", "DBF", "MF", "CSH", "OSH",
    "WSA", "SAV", "GRA", "WET", "CRO", "CVM",
]
LC_NONVEG_COLS = ["SNO", "BAR", "URB", "Water"]

DROP_Y_EQ_0 = True
DROP_ALL_MW_MISSING = True

SPLIT_BLOCK_DEG = 0.5
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
RANDOM_SEED = 42

IGBP_TO_LC_DOM = {
    "Evergreen Needleleaf Forests": "ENF",
    "Evergreen Broadleaf Forests": "EBF",
    "Deciduous Needleleaf Forests": "DNF",
    "Deciduous Broadleaf Forests": "DBF",
    "Mixed Forests": "MF",
    "Closed Shrublands": "CSH",
    "Open Shrublands": "OSH",
    "Woody Savannas": "WSA",
    "Savannas": "SAV",
    "Grasslands": "GRA",
    "Permanent Wetlands": "WET",
    "Croplands": "CRO",
    "Cropland/Natural Vegetation Mosaics": "CVM",
    "Snow and Ice": "SNO",
    "Barren": "BAR",
    "Urban and Built-up Lands": "URB",
    "Water Bodies": "Water",
}


# =======================
# Metrics / helpers
# =======================
def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(skm.mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R": r}


def spatial_block_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()

    out["_block_lat"] = np.floor((out[LAT_COL] + 90.0) / SPLIT_BLOCK_DEG).astype(int)
    out["_block_lon"] = np.floor((out[LON_COL] + 180.0) / SPLIT_BLOCK_DEG).astype(int)
    out["block_id"] = out["_block_lat"].astype(str) + "_" + out["_block_lon"].astype(str)

    blocks = out["block_id"].drop_duplicates().to_numpy()
    rng.shuffle(blocks)

    n_blocks = len(blocks)
    n_train = int(round(n_blocks * TRAIN_FRAC))
    n_val = int(round(n_blocks * VAL_FRAC))

    train_blocks = set(blocks[:n_train])
    val_blocks = set(blocks[n_train:n_train + n_val])
    test_blocks = set(blocks[n_train + n_val:])

    out["split"] = "drop"
    out.loc[out["block_id"].isin(train_blocks), "split"] = "train"
    out.loc[out["block_id"].isin(val_blocks), "split"] = "val"
    out.loc[out["block_id"].isin(test_blocks), "split"] = "test"

    return out[out["split"].isin(["train", "val", "test"])].copy()


def _canonical_igbp_name(prop_col: str) -> str:
    return prop_col.replace("IGBP_prop_", "").replace("_", " ")


def _build_igbp_prop_columns(columns: Iterable[str]) -> Tuple[List[str], List[str]]:
    all_prop_cols = [c for c in columns if c.startswith("IGBP_prop_")]
    nonveg = {"Barren", "Urban and Built-up Lands", "Water Bodies", "Snow and Ice"}
    veg_prop_cols = [c for c in all_prop_cols if _canonical_igbp_name(c) not in nonveg]
    return all_prop_cols, veg_prop_cols


def _safe_prop_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    mat = np.nan_to_num(
        df[cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    mat = np.clip(mat, 0.0, None)
    return mat


# =======================
# Load / preprocess
# =======================
def load_base_dataframe(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    all_igbp_prop_cols, _ = _build_igbp_prop_columns(df.columns)

    need = [DATE_COL, LAT_COL, LON_COL, Y_COL, FT_COL, IGBP_COL] + VOD_COLS + CORE_NUM_COLS_LITE + LC_VEG_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = VOD_COLS + CORE_NUM_COLS_LITE + LC_VEG_COLS + LC_NONVEG_COLS + all_igbp_prop_cols
    numeric_cols_present = [c for c in numeric_cols if c in df.columns]

    for col in [Y_COL, LAT_COL, LON_COL] + numeric_cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.floor("D")

    df = df[
        df[Y_COL].notna()
        & df[LAT_COL].notna()
        & df[LON_COL].notna()
        & df[DATE_COL].notna()
    ].copy()

    if DROP_Y_EQ_0:
        df = df[df[Y_COL] != 0].copy()

    if DROP_ALL_MW_MISSING:
        all_mw_missing = df[VOD_COLS].isna().all(axis=1)
        df = df.loc[~all_mw_missing].copy()

    strict_core_cols = [c for c in CORE_NUM_COLS_LITE if c in df.columns]
    if strict_core_cols:
        df = df.dropna(subset=strict_core_cols).copy()

    doy = df[DATE_COL].dt.dayofyear.astype(np.float32)
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)
    df["lat_norm"] = (df[LAT_COL] / 90.0).astype(np.float32)

    lc_mat = _safe_prop_matrix(df, LC_VEG_COLS)
    lc_idx = np.argmax(lc_mat, axis=1)

    lc_sum = lc_mat.sum(axis=1, keepdims=True)
    lc_sum[lc_sum == 0] = np.nan
    lc_norm = lc_mat / lc_sum

    lc_names = np.array(LC_VEG_COLS, dtype=object)
    df["lc_dom"] = pd.Categorical(lc_names[lc_idx], categories=LC_VEG_COLS)
    df["lc_dom_frac"] = lc_norm[np.arange(lc_norm.shape[0]), lc_idx].astype(np.float32)

    df["igbp_to_lc_dom"] = df[IGBP_COL].astype("string").map(IGBP_TO_LC_DOM).fillna("Unknown")

    return df


# =======================
# Final S6-lite dataset
# =======================
def build_final_s6_lite_frame(base_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
    df = base_df.copy()

    # Keep S6 consistency filter
    df = df[df["igbp_to_lc_dom"] == df["lc_dom"].astype("string")].copy()

    # S6-lite numeric inputs:
    # - keep VOD
    # - keep LAI / Hveg / LST
    # - keep doy_sin / doy_cos / lat_norm
    # - drop SM
    # - drop lc_dom_frac / lc_entropy / weights / IGBP_prop_*
    num_cols = VOD_COLS + CORE_NUM_COLS_LITE + SEASON_COLS
    cat_col = "lc_dom"

    use_cols = [DATE_COL, LAT_COL, LON_COL, Y_COL, cat_col] + num_cols
    df = df[use_cols].copy()

    df = df[df[Y_COL].notna()].copy()
    df = df[df[cat_col].notna()].copy()

    # Allow partial VOD missingness, but not all missing
    df = df[~df[VOD_COLS].isna().all(axis=1)].copy()

    strict_num_cols = [c for c in num_cols if c not in VOD_COLS]
    if strict_num_cols:
        df = df.dropna(subset=strict_num_cols).copy()

    # No sample weights in S6-lite
    df["sample_weight"] = 1.0

    df = spatial_block_split(df, seed=RANDOM_SEED)
    return df, num_cols, cat_col


# =======================
# Result helpers
# =======================
def summarize_results(rows: List[Dict]):
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["scheme", "model", "split"], as_index=False)[["MAE", "RMSE", "R"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in summary.columns.to_flat_index()
    ]
    return raw, summary


def save_result_tables(rows: List[Dict], out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, summary = summarize_results(rows)
    raw_path = out_dir / f"{stem}_raw.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return raw_path, summary_path
