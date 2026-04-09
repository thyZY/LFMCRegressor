from __future__ import annotations

from dataclasses import dataclass
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
OUT_DIR = Path(r"d:\Python\jupyter\jupyter\LFMCRegressor\figs\batch_runs_ablation")

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
CORE_NUM_COLS = ["LAI", "Hveg", "SM", "LST"]
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

WEIGHT_MIN = 0.2
WEIGHT_GAMMA = 2.0

NONVEG_THRESHOLD = 0.05

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


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(skm.mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R": r}


def purity_to_weight(
    purity: np.ndarray,
    gamma: float = WEIGHT_GAMMA,
    w_min: float = WEIGHT_MIN,
) -> np.ndarray:
    purity = np.asarray(purity, dtype=np.float32)
    purity = np.nan_to_num(purity, nan=0.0, posinf=1.0, neginf=0.0)
    purity = np.clip(purity, 0.0, 1.0)

    w = np.maximum(w_min, purity ** gamma).astype(np.float32)
    w = np.nan_to_num(w, nan=w_min, posinf=1.0, neginf=w_min)
    w[w < 0] = w_min
    return w


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


def _calc_entropy(mat: np.ndarray) -> np.ndarray:
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = np.nan
    norm = mat / row_sum
    eps = 1e-12
    return (-np.nansum(norm * np.log(np.clip(norm, eps, 1.0)), axis=1)).astype(np.float32)


def _safe_prop_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    mat = np.nan_to_num(
        df[cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    mat = np.clip(mat, 0.0, None)
    return mat


def _build_igbp_from_single_label(df: pd.DataFrame) -> pd.DataFrame:
    igbp_label = df[IGBP_COL].astype("string").fillna("Unknown")
    cats = sorted([c for c in igbp_label.dropna().unique().tolist() if c != "Unknown"])
    if not cats:
        cats = ["Unknown"]

    df["igbp_dom"] = pd.Categorical(igbp_label, categories=cats)
    df["igbp_dom_frac"] = np.where(igbp_label == "Unknown", 0.0, 1.0).astype(np.float32)
    df["igbp_entropy"] = np.where(igbp_label == "Unknown", np.nan, 0.0).astype(np.float32)
    df["igbp_weight"] = purity_to_weight(df["igbp_dom_frac"].to_numpy())
    return df


def load_base_dataframe(data_path: Path | None = None) -> Tuple[pd.DataFrame, List[str]]:
    path = data_path or DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    all_igbp_prop_cols, igbp_veg_prop_cols = _build_igbp_prop_columns(df.columns)

    need = [DATE_COL, LAT_COL, LON_COL, Y_COL, FT_COL, IGBP_COL] + VOD_COLS + CORE_NUM_COLS + LC_VEG_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = VOD_COLS + CORE_NUM_COLS + LC_VEG_COLS + LC_NONVEG_COLS + all_igbp_prop_cols
    numeric_cols_present = [c for c in numeric_cols if c in df.columns]

    for col in [Y_COL, LAT_COL, LON_COL] + numeric_cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.floor("D")
    df = df[df[Y_COL].notna() & df[LAT_COL].notna() & df[LON_COL].notna() & df[DATE_COL].notna()].copy()

    if DROP_Y_EQ_0:
        df = df[df[Y_COL] != 0].copy()

    if DROP_ALL_MW_MISSING:
        all_mw_missing = df[VOD_COLS].isna().all(axis=1)
        df = df.loc[~all_mw_missing].copy()

    strict_core_cols = [c for c in CORE_NUM_COLS if c in df.columns]
    if strict_core_cols:
        df = df.dropna(subset=strict_core_cols).copy()

    doy = df[DATE_COL].dt.dayofyear.astype(np.float32)
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)
    df["lat_norm"] = (df[LAT_COL] / 90.0).astype(np.float32)

    if igbp_veg_prop_cols:
        igbp_mat = _safe_prop_matrix(df, igbp_veg_prop_cols)
        igbp_idx = np.argmax(igbp_mat, axis=1)

        igbp_names = np.array([_canonical_igbp_name(c) for c in igbp_veg_prop_cols], dtype=object)

        igbp_sum = igbp_mat.sum(axis=1, keepdims=True)
        igbp_sum[igbp_sum == 0] = np.nan
        igbp_norm = igbp_mat / igbp_sum

        df["igbp_dom"] = pd.Categorical(igbp_names[igbp_idx], categories=list(igbp_names))
        df["igbp_dom_frac"] = igbp_norm[np.arange(igbp_norm.shape[0]), igbp_idx].astype(np.float32)
        df["igbp_entropy"] = _calc_entropy(igbp_mat)
        df["igbp_weight"] = purity_to_weight(df["igbp_dom_frac"].to_numpy())
    else:
        df = _build_igbp_from_single_label(df)

    lc_mat = _safe_prop_matrix(df, LC_VEG_COLS)
    lc_idx = np.argmax(lc_mat, axis=1)

    lc_sum = lc_mat.sum(axis=1, keepdims=True)
    lc_sum[lc_sum == 0] = np.nan
    lc_norm = lc_mat / lc_sum

    lc_names = np.array(LC_VEG_COLS, dtype=object)
    df["lc_dom"] = pd.Categorical(lc_names[lc_idx], categories=LC_VEG_COLS)
    df["lc_dom_frac"] = lc_norm[np.arange(lc_norm.shape[0]), lc_idx].astype(np.float32)
    df["lc_entropy"] = _calc_entropy(lc_mat)
    df["lc_weight"] = purity_to_weight(df["lc_dom_frac"].to_numpy())

    nonveg_cols_present = [c for c in LC_NONVEG_COLS if c in df.columns]
    if nonveg_cols_present:
        nonveg = np.nan_to_num(
            df[nonveg_cols_present].to_numpy(dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        nonveg = np.clip(nonveg, 0.0, None).sum(axis=1)
        df["nonveg_frac"] = np.where(nonveg > 1.0, nonveg / 100.0, nonveg).astype(np.float32)
    else:
        df["nonveg_frac"] = 0.0

    df["igbp_to_lc_dom"] = df[IGBP_COL].astype("string").map(IGBP_TO_LC_DOM).fillna("Unknown")
    igbp_num_cols = igbp_veg_prop_cols.copy()
    return df, igbp_num_cols


@dataclass(frozen=True)
class AblationSpecV2:
    code: str
    description: str
    category_col: str = "lc_dom"
    include_vod: bool = True
    include_core: bool = True
    include_season: bool = True
    include_igbp_props: bool = True
    extra_num_cols: Tuple[str, ...] = ("lc_dom_frac", "lc_entropy")
    weight_col: str | None = "lc_weight"
    apply_consistency: bool = True
    apply_nonveg_filter: bool = False
    exclude_num_cols: Tuple[str, ...] = tuple()


ABLATIONS_V2: List[AblationSpecV2] = [
    AblationSpecV2(
        code="FULL_S6",
        description="exact S6 baseline: lc_dom + frac + entropy + weight + consistency",
    ),
    AblationSpecV2(
        code="B1_NO_DOY_SIN",
        description="remove doy_sin only",
        exclude_num_cols=("doy_sin",),
    ),
    AblationSpecV2(
        code="B2_NO_DOY_COS",
        description="remove doy_cos only",
        exclude_num_cols=("doy_cos",),
    ),
    AblationSpecV2(
        code="B3_NO_LAT_NORM",
        description="remove lat_norm only",
        exclude_num_cols=("lat_norm",),
    ),
    AblationSpecV2(
        code="B4_NO_LAI",
        description="remove LAI only",
        exclude_num_cols=("LAI",),
    ),
    AblationSpecV2(
        code="B5_NO_SM",
        description="remove SM only",
        exclude_num_cols=("SM",),
    ),
    AblationSpecV2(
        code="B6_NO_LST",
        description="remove LST only",
        exclude_num_cols=("LST",),
    ),
    AblationSpecV2(
        code="B7_NO_HVEG",
        description="remove Hveg only",
        exclude_num_cols=("Hveg",),
    ),
    AblationSpecV2(
        code="NO_CONS",
        description="remove consistency filter only",
        apply_consistency=False,
    ),
]


def build_ablation_v2_frame(
    base_df: pd.DataFrame,
    spec: AblationSpecV2,
    igbp_num_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], str]:
    df = base_df.copy()

    if spec.apply_consistency:
        df = df[df["igbp_to_lc_dom"] == df["lc_dom"].astype("string")].copy()

    if spec.apply_nonveg_filter:
        df = df[df["nonveg_frac"] < NONVEG_THRESHOLD].copy()

    fixed_num_cols: List[str] = []
    if spec.include_vod:
        fixed_num_cols += VOD_COLS
    if spec.include_core:
        fixed_num_cols += CORE_NUM_COLS
    if spec.include_season:
        fixed_num_cols += SEASON_COLS
    if spec.include_igbp_props:
        fixed_num_cols += igbp_num_cols

    num_cols = fixed_num_cols + list(spec.extra_num_cols)
    num_cols = [c for c in num_cols if c not in set(spec.exclude_num_cols)]

    cat_col = spec.category_col

    use_cols = [DATE_COL, LAT_COL, LON_COL, Y_COL, cat_col] + num_cols
    if spec.weight_col is not None:
        use_cols.append(spec.weight_col)

    use_cols = list(dict.fromkeys(use_cols))
    df = df[use_cols].copy()

    df = df[df[Y_COL].notna()].copy()
    df = df[df[cat_col].notna()].copy()

    if spec.include_vod:
        df = df[~df[VOD_COLS].isna().all(axis=1)].copy()

    strict_num_cols = [c for c in num_cols if c not in VOD_COLS]
    if strict_num_cols:
        df = df.dropna(subset=strict_num_cols).copy()

    if spec.weight_col is None:
        df["sample_weight"] = 1.0
    else:
        w = pd.to_numeric(df[spec.weight_col], errors="coerce").astype(np.float32)
        w = np.nan_to_num(w, nan=WEIGHT_MIN, posinf=1.0, neginf=WEIGHT_MIN)
        w = np.clip(w, WEIGHT_MIN, None)
        df["sample_weight"] = w

    df = spatial_block_split(df, seed=RANDOM_SEED)
    return df, num_cols, cat_col


def summarize_results(rows: List[Dict]):
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["ablation", "model", "split"], as_index=False)[["MAE", "RMSE", "R"]]
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
