# -*- coding: utf-8 -*-
"""
common_features.py

用途：
1) 保留原有 CSV 方式读取基础表
2) 新增 raster 模式：直接从原始变量文件构建单日全球 0.1° 预测表
3) 输出给产品预测脚本使用的 base_df 和 meta

说明：
- 网格固定为 1800 x 3600，0.1°
- 输入变量路径和命名规则直接复用训练样本准备阶段的逻辑
- 当前会构造：
    * row / col
    * lat / lon
    * date
    * Hveg
    * VOD 6波段
    * SM / VOD_QC
    * LAI
    * LST
    * Rainfall / Rainfall_7day / Rainfall_7day_valid_days
    * LC 17类比例
    * doy / doy_sin / doy_cos / lat_norm
    * lc_dom / lc_dom_frac / nonveg_frac / nonveg_skip
- 若 require_target=False，不要求 LFMC 目标列存在
"""

import os
import gc
import json
import math
import h5py
import warnings
import logging
from bisect import bisect_left
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
except Exception:
    sio = None

warnings.filterwarnings("ignore")

# ============================== 日志 ==============================
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# ============================== 路径配置 ==============================
CSV_DEFAULT_PATH = r"G:\data\Globe LFMC\country_statistics\globe_lfmc_filtered_step3_daily_0p1deg_with_predictors.csv"

VOD_BASE_PATH = r"G:\data\VOD\AMSR-VOD\MCCA-VOD(KuCX)\Asc\01h5"
LAI_8DAY_BASE_PATH = r"G:\data\GLASS LAI\01nc"
LST_BASE_PATH = r"G:\data\LST_yupei\01h5"
RAIN_BASE_PATH = r"G:\data\GRM rainfall\mat"
LC_BASE_PATH = r"G:\data\MCD12C1 CMG\01Degree\mat"
HVEG_PATH = r"G:\data\CanopyHeight\CH.mat"

# ============================== 列名配置 ==============================
DATE_COL = "Sampling date (YYYYMMDD)"
LAT_COL = "Latitude (WGS84, EPSG:4326)"
LON_COL = "Longitude (WGS84, EPSG:4326)"
TARGET_COL = "LFMC value (%)"

DATE_FORMAT = None

# ============================== 网格常量 ==============================
ROWS = 1800
COLS = 3600
TARGET_SHAPE = (ROWS, COLS)

CELL_SIZE = 0.1
ORIGIN_X = -180.0
ORIGIN_Y = 90.0

# ============================== 原始变量配置 ==============================
VOD_FEATURES = [
    ("VOD_Ku_Hpol_Asc", "ku_vod_H"),
    ("VOD_Ku_Vpol_Asc", "ku_vod_V"),
    ("VOD_X_Hpol_Asc", "x_vod_H"),
    ("VOD_X_Vpol_Asc", "x_vod_V"),
    ("VOD_C_Hpol_Asc", "c_vod_H"),
    ("VOD_C_Vpol_Asc", "c_vod_V"),
]

SM_FEATURE = ("SM", "SM")

LC_FEATURES = [
    ("ENF", "ENF"),
    ("EBF", "EBF"),
    ("DNF", "DNF"),
    ("DBF", "DBF"),
    ("MF", "MF"),
    ("CSH", "CSH"),
    ("OSH", "OSH"),
    ("WSA", "WSA"),
    ("SAV", "SAV"),
    ("GRA", "GRA"),
    ("WET", "WET"),
    ("CRO", "CRO"),
    ("CVM", "CVM"),
    ("SNO", "SNO"),
    ("BAR", "BAR"),
    ("URB", "URB"),
    ("Water", "Water"),
]

LC_KEY_ALIASES = {
    "ENF": ["ENF"],
    "EBF": ["EBF"],
    "DNF": ["DNF"],
    "DBF": ["DBF"],
    "MF": ["MF"],
    "CSH": ["CSH"],
    "OSH": ["OSH"],
    "WSA": ["WSA"],
    "SAV": ["SAV"],
    "GRA": ["GRA"],
    "WET": ["WET"],
    "CRO": ["CRO"],
    "CVM": ["CVM"],
    "SNO": ["SNO", "Snow", "SnowIce", "PermanentSnowIce", "Permanent Snow and Ice"],
    "BAR": ["BAR", "Barren"],
    "URB": ["URB", "Urban", "UrbanBuiltup", "Urban_Builtup", "Urban and Builtup Lands"],
    "Water": ["Water", "WAT", "WaterBodies", "Water Bodies"],
}

# ============================== 缓存 ==============================
HVEG_DATA = None
LC_YEAR_CACHE = {}
LAI_8DAY_AVAILABLE_DATES = None


# =============================================================================
# 基础工具函数
# =============================================================================
def parse_date(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, datetime):
        return pd.Timestamp(x)
    if DATE_FORMAT is None:
        return pd.to_datetime(x)
    return pd.to_datetime(x, format=DATE_FORMAT)


def latlon_to_rowcol(lat, lon):
    """
    geotransform = (-180, 0.1, 0, 90, 0, -0.1)
    row = floor((90 - lat) / 0.1)
    col = floor((lon + 180) / 0.1)
    """
    if pd.isna(lat) or pd.isna(lon):
        return -1, -1

    row = int(np.floor((ORIGIN_Y - lat) / CELL_SIZE))
    col = int(np.floor((lon - ORIGIN_X) / CELL_SIZE))

    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return -1, -1

    return row, col


def rowcol_to_latlon(rows, cols):
    """
    使用像元中心点坐标
    lat = 90 - (row + 0.5) * 0.1
    lon = -180 + (col + 0.5) * 0.1
    """
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    lat = ORIGIN_Y - (rows + 0.5) * CELL_SIZE
    lon = ORIGIN_X + (cols + 0.5) * CELL_SIZE
    return lat.astype(np.float32), lon.astype(np.float32)


def build_global_grid():
    rows = np.repeat(np.arange(ROWS, dtype=np.int32), COLS)
    cols = np.tile(np.arange(COLS, dtype=np.int32), ROWS)
    lat, lon = rowcol_to_latlon(rows, cols)
    return rows, cols, lat, lon


def first_existing_dataset(f, preferred_keys=None):
    preferred_keys = preferred_keys or []
    for key in preferred_keys:
        if key in f:
            return key
    keys = list(f.keys())
    if len(keys) == 0:
        return None
    return keys[0]


def ensure_2d_numpy_1800x3600(arr, name="array", force_float32=True):
    """
    输入为 numpy 数组：
    - 若是 (1800, 3600) 直接返回
    - 若是 (3600, 1800) 则转置
    - 其他形状报错
    """
    arr = np.asarray(arr)

    if arr.shape == (COLS, ROWS):
        arr = arr.T
    elif arr.shape != (ROWS, COLS):
        raise ValueError(f"{name} shape 不支持: {arr.shape}")

    if force_float32 and arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    return arr


def extract_values_by_rowcol(arr, rows, cols, out_dtype=np.float32):
    """
    从 numpy 2D 数组中按 rows/cols 批量取值
    """
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)

    if np.issubdtype(np.dtype(out_dtype), np.floating):
        out = np.full(len(rows), np.nan, dtype=out_dtype)
    else:
        out = np.zeros(len(rows), dtype=out_dtype)

    valid = (
        (rows >= 0) & (rows < ROWS) &
        (cols >= 0) & (cols < COLS)
    )
    if np.any(valid):
        out[valid] = arr[rows[valid], cols[valid]]

    return out


def safe_float32(x):
    arr = np.asarray(x, dtype=np.float32)
    return arr


# =============================================================================
# Hveg
# =============================================================================
def load_hveg():
    global HVEG_DATA

    if HVEG_DATA is not None:
        return HVEG_DATA

    if not os.path.exists(HVEG_PATH):
        logger.error(f"Hveg 文件不存在: {HVEG_PATH}")
        return None

    try:
        with h5py.File(HVEG_PATH, "r") as f:
            key = first_existing_dataset(f, ["Hveg", "CanopyHeight", "CH"])
            if key is None:
                logger.error("Hveg 文件中未找到数据集")
                return None

            arr = np.asarray(f[key][:], dtype=np.float32)
            arr = ensure_2d_numpy_1800x3600(arr, f"Hveg:{key}", force_float32=True)
            HVEG_DATA = arr

        logger.info(f"Hveg 加载成功: {key}, shape={HVEG_DATA.shape}")
        return HVEG_DATA

    except Exception as e:
        logger.error(f"Hveg 加载失败: {str(e)}", exc_info=True)
        return None


# =============================================================================
# VOD / SM / QC
# =============================================================================
def get_vod_file(date):
    fp = os.path.join(VOD_BASE_PATH, date.strftime("%Y%m%d") + ".h5")
    return fp if os.path.exists(fp) else None


def extract_vod_sm_qc_for_date(date, rows, cols):
    out = {name: np.full(len(rows), np.nan, dtype=np.float32) for name, _ in VOD_FEATURES}
    out["SM"] = np.full(len(rows), np.nan, dtype=np.float32)
    out["VOD_QC"] = np.full(len(rows), np.nan, dtype=np.float32)

    vod_file = get_vod_file(date)
    if vod_file is None:
        logger.warning(f"VOD 文件不存在: {date.strftime('%Y-%m-%d')}")
        return out

    try:
        with h5py.File(vod_file, "r") as f:
            file_keys = set(f.keys())

            qc_key = None
            for cand in ["QC", "qc"]:
                if cand in file_keys:
                    qc_key = cand
                    break

            if qc_key is None:
                logger.warning(f"VOD 文件中未找到 QC: {vod_file}")
                return out

            try:
                qc_arr = np.asarray(f[qc_key][:])
                qc_arr = ensure_2d_numpy_1800x3600(qc_arr, f"QC:{qc_key}", force_float32=False)
                qc_vals = extract_values_by_rowcol(qc_arr, rows, cols, out_dtype=np.uint8)
                out["VOD_QC"] = qc_vals.astype(np.float32)
                valid_qc_mask = (qc_vals == 0)
            except Exception as e_qc:
                logger.error(f"QC 读取失败: {vod_file} | {qc_key} - {str(e_qc)}", exc_info=True)
                return out
            finally:
                if "qc_arr" in locals():
                    del qc_arr

            if not np.any(valid_qc_mask):
                return out

            for out_name, source_key in VOD_FEATURES:
                found_key = None
                if source_key in file_keys:
                    found_key = source_key
                elif source_key.lower() in file_keys:
                    found_key = source_key.lower()
                elif out_name in file_keys:
                    found_key = out_name

                if found_key is None:
                    logger.warning(f"VOD 变量缺失: {out_name}，保留 NaN")
                    continue

                try:
                    arr = np.asarray(f[found_key][:], dtype=np.float32)
                    arr = ensure_2d_numpy_1800x3600(arr, f"VOD:{found_key}", force_float32=True)
                    vals = extract_values_by_rowcol(arr, rows, cols)
                    vals[~valid_qc_mask] = np.nan
                    out[out_name] = vals
                except Exception as e_var:
                    logger.error(f"VOD 变量读取失败: {vod_file} | {found_key} - {str(e_var)}", exc_info=True)
                finally:
                    if "arr" in locals():
                        del arr

            sm_found_key = None
            for cand in ["SM", "sm", "soil_moisture", "SoilMoisture"]:
                if cand in file_keys:
                    sm_found_key = cand
                    break

            if sm_found_key is None:
                logger.warning(f"VOD 文件中未找到 SM 变量: {vod_file}")
            else:
                try:
                    sm_arr = np.asarray(f[sm_found_key][:], dtype=np.float32)
                    sm_arr = ensure_2d_numpy_1800x3600(sm_arr, f"SM:{sm_found_key}", force_float32=True)
                    sm_vals = extract_values_by_rowcol(sm_arr, rows, cols)
                    sm_vals[~valid_qc_mask] = np.nan
                    out["SM"] = sm_vals
                except Exception as e_sm:
                    logger.error(f"SM 变量读取失败: {vod_file} | {sm_found_key} - {str(e_sm)}", exc_info=True)
                finally:
                    if "sm_arr" in locals():
                        del sm_arr

        return out

    except Exception as e:
        logger.error(f"VOD/SM/QC 读取失败: {vod_file} - {str(e)}", exc_info=True)
        return out


# =============================================================================
# LAI 8-day
# =============================================================================
def get_all_lai_8day_dates():
    global LAI_8DAY_AVAILABLE_DATES

    if LAI_8DAY_AVAILABLE_DATES is not None:
        return LAI_8DAY_AVAILABLE_DATES

    available_dates = []

    if not os.path.exists(LAI_8DAY_BASE_PATH):
        logger.error(f"LAI 8日目录不存在: {LAI_8DAY_BASE_PATH}")
        LAI_8DAY_AVAILABLE_DATES = []
        return LAI_8DAY_AVAILABLE_DATES

    for fname in os.listdir(LAI_8DAY_BASE_PATH):
        if fname.lower().endswith(".h5"):
            date_str = os.path.splitext(fname)[0]
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                available_dates.append(dt)
            except Exception:
                pass

    available_dates = sorted(available_dates)
    LAI_8DAY_AVAILABLE_DATES = available_dates
    logger.info(f"8日 LAI 可用日期数: {len(available_dates)}")
    return LAI_8DAY_AVAILABLE_DATES


def get_lai_8day_file(date):
    fp = os.path.join(LAI_8DAY_BASE_PATH, date.strftime("%Y%m%d") + ".h5")
    return fp if os.path.exists(fp) else None


def extract_lai_points_from_file(file_path, rows, cols):
    out = np.full(len(rows), np.nan, dtype=np.float32)

    if file_path is None or not os.path.exists(file_path):
        return out

    try:
        with h5py.File(file_path, "r") as f:
            if "LAI" not in f:
                logger.error(f"8日 LAI 文件缺少 LAI 变量: {file_path}")
                return out

            arr = np.asarray(f["LAI"][:], dtype=np.float32)
            arr = ensure_2d_numpy_1800x3600(arr, "LAI", force_float32=True)
            out = extract_values_by_rowcol(arr, rows, cols)

            del arr
            return out

    except Exception as e:
        logger.error(f"读取 8日 LAI 失败: {file_path} - {str(e)}", exc_info=True)
        return out


def extract_lai_for_date(date, rows, cols):
    """
    只使用 8日 LAI：
    1) 当天有文件 -> 直接按点读取
    2) 否则读取前后最近日期并按点线性插值
    """
    direct_file = get_lai_8day_file(date)
    if direct_file is not None:
        return extract_lai_points_from_file(direct_file, rows, cols)

    available_dates = get_all_lai_8day_dates()
    if not available_dates:
        logger.warning("没有可用的 8日 LAI 日期")
        return np.full(len(rows), np.nan, dtype=np.float32)

    idx = bisect_left(available_dates, date)

    prev_date = available_dates[idx - 1] if idx > 0 else None
    next_date = available_dates[idx] if idx < len(available_dates) else None

    if prev_date is None or next_date is None:
        logger.warning(f"LAI 无法对 {date.strftime('%Y-%m-%d')} 进行前后插值")
        return np.full(len(rows), np.nan, dtype=np.float32)

    prev_vals = extract_lai_points_from_file(get_lai_8day_file(prev_date), rows, cols)
    next_vals = extract_lai_points_from_file(get_lai_8day_file(next_date), rows, cols)

    if np.all(np.isnan(prev_vals)) and np.all(np.isnan(next_vals)):
        logger.warning(
            f"LAI 插值端点读取失败: prev={prev_date.strftime('%Y-%m-%d')}, "
            f"next={next_date.strftime('%Y-%m-%d')}"
        )
        return np.full(len(rows), np.nan, dtype=np.float32)

    total_days = (next_date - prev_date).days
    offset_days = (date - prev_date).days

    if total_days == 0:
        return prev_vals.astype(np.float32)

    w = offset_days / total_days
    return (prev_vals * (1.0 - w) + next_vals * w).astype(np.float32)


# =============================================================================
# LST
# =============================================================================
def get_lst_file(date):
    fp = os.path.join(LST_BASE_PATH, date.strftime("%Y%m%d") + ".h5")
    return fp if os.path.exists(fp) else None


def extract_lst_for_date(date, rows, cols, var_name="LST_Day"):
    out = np.full(len(rows), np.nan, dtype=np.float32)

    lst_file = get_lst_file(date)
    if lst_file is None:
        logger.warning(f"LST 文件不存在: {date.strftime('%Y-%m-%d')}")
        return out

    try:
        with h5py.File(lst_file, "r") as f:
            if var_name not in f:
                logger.warning(f"LST 文件中未找到变量 {var_name}: {lst_file}")
                return out

            arr = np.asarray(f[var_name][:], dtype=np.float32)
            arr = ensure_2d_numpy_1800x3600(arr, f"LST:{var_name}", force_float32=True)
            out = extract_values_by_rowcol(arr, rows, cols)

            del arr
            return out

    except Exception as e:
        logger.error(f"LST 读取失败: {lst_file} - {str(e)}", exc_info=True)
        return out


# =============================================================================
# Rainfall
# =============================================================================
def get_rain_mat_file(date):
    fp = os.path.join(RAIN_BASE_PATH, date.strftime("%Y%m%d") + ".mat")
    return fp if os.path.exists(fp) else None


def get_rain_nc4_file(date):
    fn = f"3B-DAY.MS.MRG.3IMERG.{date.strftime('%Y%m%d')}-S000000-E235959.V07B.nc4"
    fp = os.path.join(RAIN_BASE_PATH, fn)
    return fp if os.path.exists(fp) else None


def load_rain_from_mat(mat_file, var_name="Precipitation"):
    try:
        with h5py.File(mat_file, "r") as f:
            if var_name not in f:
                raise KeyError(f"mat(v7.3) 中未找到变量 {var_name}")
            arr = np.asarray(f[var_name][:], dtype=np.float32)
            arr = ensure_2d_numpy_1800x3600(arr, f"Rainfall-mat:{var_name}", force_float32=True)
            arr[arr < 0] = np.nan
            return arr
    except Exception as e_h5:
        logger.warning(f"Rainfall mat 按 v7.3 读取失败，尝试旧版 mat: {mat_file} | {str(e_h5)}")

    if sio is None:
        raise RuntimeError("scipy.io 不可用，无法读取非 v7.3 mat 文件")

    data = sio.loadmat(mat_file)
    if var_name not in data:
        raise KeyError(f"mat(old) 中未找到变量 {var_name}: {mat_file}")

    arr = np.asarray(data[var_name], dtype=np.float32)
    arr = ensure_2d_numpy_1800x3600(arr, f"Rainfall-mat-old:{var_name}", force_float32=True)
    arr[arr < 0] = np.nan
    return arr


def load_rain_from_nc4(nc4_file, var_name="precipitation"):
    with h5py.File(nc4_file, "r") as f:
        if var_name not in f:
            raise KeyError(f"nc4 中未找到变量 {var_name}: {nc4_file}")

        ds = f[var_name]
        if ds.ndim == 3:
            arr = np.asarray(ds[0, :, :], dtype=np.float32)
        elif ds.ndim == 2:
            arr = np.asarray(ds[:, :], dtype=np.float32)
        else:
            raise ValueError(f"Rainfall nc4 变量维度不支持: {ds.shape}")

    arr = ensure_2d_numpy_1800x3600(arr, f"Rainfall-nc4:{var_name}", force_float32=True)
    arr[arr < 0] = np.nan
    return arr


def load_rain_array_for_date(date):
    mat_file = get_rain_mat_file(date)
    nc4_file = get_rain_nc4_file(date)

    if mat_file is not None:
        try:
            return load_rain_from_mat(mat_file, var_name="Precipitation")
        except Exception as e_mat:
            logger.warning(f"Rainfall mat 读取失败: {mat_file} | {str(e_mat)}")
            if nc4_file is None:
                raise

    if nc4_file is not None:
        try:
            return load_rain_from_nc4(nc4_file, var_name="precipitation")
        except Exception as e_nc:
            logger.error(f"Rainfall nc4 读取失败: {nc4_file} | {str(e_nc)}", exc_info=True)
            raise

    raise FileNotFoundError(f"Rainfall 文件不存在: {date.strftime('%Y-%m-%d')}")


def extract_rainfall_for_date(date, rows, cols):
    out = np.full(len(rows), np.nan, dtype=np.float32)

    try:
        arr = load_rain_array_for_date(date)
        out = extract_values_by_rowcol(arr, rows, cols)
        del arr
        return out
    except Exception as e:
        logger.warning(f"Rainfall 单日读取失败: {date.strftime('%Y-%m-%d')} | {str(e)}")
        return out


def extract_rainfall_7day_for_date(date, rows, cols, include_current=True, days=7):
    vals_list = []

    if include_current:
        start_offset = days - 1
        date_list = [date - timedelta(days=i) for i in range(start_offset, -1, -1)]
    else:
        date_list = [date - timedelta(days=i) for i in range(days, 0, -1)]

    for d in date_list:
        vals = extract_rainfall_for_date(d, rows, cols)
        vals_list.append(vals)

    stack = np.vstack(vals_list).astype(np.float32)
    valid_days = np.sum(~np.isnan(stack), axis=0).astype(np.uint8)
    out_sum = np.nansum(stack, axis=0).astype(np.float32)

    all_nan_mask = (valid_days == 0)
    out_sum[all_nan_mask] = np.nan

    del stack, vals_list
    return out_sum, valid_days


# =============================================================================
# LC
# =============================================================================
def get_lc_file(year):
    possible_files = [
        f"{year}001.mat",
        f"{year}.mat",
        f"LC_{year}.mat",
        f"MCD12C1_{year}001.mat"
    ]

    for fn in possible_files:
        fp = os.path.join(LC_BASE_PATH, fn)
        if os.path.exists(fp):
            return fp
    return None


def load_lc_for_year(year):
    global LC_YEAR_CACHE

    if year in LC_YEAR_CACHE:
        return LC_YEAR_CACHE[year]

    lc_file = get_lc_file(year)
    if lc_file is None:
        logger.warning(f"LC 文件不存在: {year}")
        return None

    lc_data = {}

    try:
        with h5py.File(lc_file, "r") as f:
            file_keys = set(f.keys())

            for std_key, out_col in LC_FEATURES:
                aliases = LC_KEY_ALIASES.get(std_key, [std_key])

                found_key = None
                for alias in aliases:
                    if alias in file_keys:
                        found_key = alias
                        break

                if found_key is None:
                    logger.warning(f"LC 变量未找到: {std_key}，用 NaN 填充")
                    lc_data[out_col] = np.full(TARGET_SHAPE, np.nan, dtype=np.float32)
                    continue

                arr = np.asarray(f[found_key][:], dtype=np.float32)
                arr = ensure_2d_numpy_1800x3600(arr, f"LC:{found_key}", force_float32=True)
                lc_data[out_col] = arr

        LC_YEAR_CACHE[year] = lc_data
        logger.info(f"LC 加载成功: {year}")
        return lc_data

    except Exception as e:
        logger.error(f"LC 读取失败: {lc_file} - {str(e)}", exc_info=True)
        return None


# =============================================================================
# 派生特征
# =============================================================================
def _safe_divide(a, b):
    out = np.full_like(a, np.nan, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[valid] = (a[valid] / b[valid]).astype(np.float32)
    return out


def add_time_and_geo_features(df):
    if DATE_COL in df.columns:
        dt = pd.to_datetime(df[DATE_COL], errors="coerce")
        doy = dt.dt.dayofyear.astype("float32")
        df["doy"] = doy
        df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.0).astype(np.float32)
        df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.0).astype(np.float32)

    if LAT_COL in df.columns:
        lat = pd.to_numeric(df[LAT_COL], errors="coerce").astype(np.float32)
        df["lat_norm"] = (lat / 90.0).astype(np.float32)

    if LON_COL in df.columns and "lon_norm" not in df.columns:
        lon = pd.to_numeric(df[LON_COL], errors="coerce").astype(np.float32)
        df["lon_norm"] = (lon / 180.0).astype(np.float32)

    return df


def add_lc_group_features(df):
    # 常用合并组
    for c in [x[1] for x in LC_FEATURES]:
        if c not in df.columns:
            df[c] = np.nan

    df["NF"] = df[["ENF", "DNF"]].sum(axis=1, min_count=1).astype(np.float32)
    df["BF"] = df[["EBF", "DBF"]].sum(axis=1, min_count=1).astype(np.float32)
    df["SH"] = df[["OSH", "CSH"]].sum(axis=1, min_count=1).astype(np.float32)
    df["Herb"] = df[["CRO", "SAV", "GRA"]].sum(axis=1, min_count=1).astype(np.float32)
    df["NonVeg"] = df[["URB", "BAR", "SNO", "Water"]].sum(axis=1, min_count=1).astype(np.float32)

    dom_cols = ["NF", "BF", "SH", "Herb", "WET", "WSA", "CVM", "MF", "NonVeg"]
    dom_arr = df[dom_cols].to_numpy(dtype=np.float32)

    valid_row = np.isfinite(dom_arr).any(axis=1)
    dom_idx = np.full(len(df), -1, dtype=np.int32)
    dom_frac = np.full(len(df), np.nan, dtype=np.float32)

    if np.any(valid_row):
        tmp = np.where(np.isfinite(dom_arr), dom_arr, -np.inf)
        dom_idx[valid_row] = np.argmax(tmp[valid_row], axis=1)
        dom_frac[valid_row] = np.nanmax(dom_arr[valid_row], axis=1)

    lc_dom = np.full(len(df), "Unknown", dtype=object)
    valid_dom = dom_idx >= 0
    lc_dom[valid_dom] = np.array(dom_cols, dtype=object)[dom_idx[valid_dom]]

    df["lc_dom"] = lc_dom
    df["lc_dom_frac"] = dom_frac.astype(np.float32)
    df["nonveg_frac"] = pd.to_numeric(df["NonVeg"], errors="coerce").astype(np.float32)
    df["nonveg_skip"] = (df["nonveg_frac"].fillna(0) >= 0.05).astype(np.uint8)

    return df


def add_predictor_valid_flag(df):
    row_ok = pd.to_numeric(df["row"], errors="coerce").fillna(-1).astype(np.int32)
    col_ok = pd.to_numeric(df["col"], errors="coerce").fillna(-1).astype(np.int32)

    df["predictor_valid"] = (
        (row_ok >= 0) & (row_ok < ROWS) &
        (col_ok >= 0) & (col_ok < COLS)
    ).astype(np.uint8)

    return df


def finalize_base_dataframe(df, require_target=True):
    df = df.copy()

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if "row" not in df.columns or "col" not in df.columns:
        if LAT_COL in df.columns and LON_COL in df.columns:
            rc = df[[LAT_COL, LON_COL]].apply(
                lambda x: latlon_to_rowcol(x[LAT_COL], x[LON_COL]), axis=1
            )
            df["row"] = [x[0] for x in rc]
            df["col"] = [x[1] for x in rc]
        else:
            raise ValueError("缺少 row/col 且无法由经纬度推导")

    df = add_time_and_geo_features(df)
    df = add_lc_group_features(df)
    df = add_predictor_valid_flag(df)

    if require_target and TARGET_COL not in df.columns:
        raise ValueError(f"缺少目标列: {TARGET_COL}")

    return df


# =============================================================================
# CSV 入口
# =============================================================================
def load_base_dataframe(path=None, require_target=True):
    """
    原 CSV 入口：
    - 读取已整理好的样本/预测表
    - 自动补充派生变量
    """
    if path is None:
        path = CSV_DEFAULT_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    return finalize_base_dataframe(df, require_target=require_target)


# =============================================================================
# Raster 入口：构造单日产品输入表
# =============================================================================
def build_prediction_dataframe_for_date(date, include_all_pixels=True):
    """
    给定某一天，直接构造整幅产品预测表

    参数
    ----
    date : str / datetime / pd.Timestamp
        例如 "2020-01-15" 或 "20200115"
    include_all_pixels : bool
        当前默认 True，返回全 648 万像元
    """
    date = parse_date(date)
    if pd.isna(date):
        raise ValueError(f"无法解析 date: {date}")

    date_py = pd.Timestamp(date).to_pydatetime()

    rows, cols, lat, lon = build_global_grid()

    df = pd.DataFrame({
        DATE_COL: pd.to_datetime(np.full(len(rows), pd.Timestamp(date_py))),
        LAT_COL: lat,
        LON_COL: lon,
        "row": rows,
        "col": cols,
    })

    logger.info(f"开始构造 {date_py.strftime('%Y-%m-%d')} 的产品输入表，共 {len(df):,} 个像元")

    # Hveg
    hveg_arr = load_hveg()
    if hveg_arr is not None:
        df["Hveg"] = extract_values_by_rowcol(hveg_arr, rows, cols)

    # VOD / SM / QC
    vod_bundle = extract_vod_sm_qc_for_date(date_py, rows, cols)
    for col_name, values in vod_bundle.items():
        df[col_name] = values

    # LAI
    df["LAI"] = extract_lai_for_date(date_py, rows, cols)

    # LST
    df["LST"] = extract_lst_for_date(date_py, rows, cols)

    # Rainfall
    df["Rainfall"] = extract_rainfall_for_date(date_py, rows, cols)
    rain7_vals, rain7_valid_days = extract_rainfall_7day_for_date(
        date_py, rows, cols, include_current=True, days=7
    )
    df["Rainfall_7day"] = rain7_vals
    df["Rainfall_7day_valid_days"] = rain7_valid_days

    # LC
    lc_data = load_lc_for_year(date_py.year)
    if lc_data is not None:
        for _std_key, out_col in LC_FEATURES:
            arr = lc_data.get(out_col)
            if arr is not None:
                df[out_col] = extract_values_by_rowcol(arr, rows, cols)

    # 派生特征
    df = finalize_base_dataframe(df, require_target=False)

    # 形成一个基础可预测标记
    critical_cols = ["LAI", "Hveg", "LST"]
    for c in critical_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 这里的 valid_for_inference 只是基础过滤，可按你的最终方案继续加严
    valid_for_inference = (
        (df["predictor_valid"] == 1) &
        (
            df[[x[0] for x in VOD_FEATURES]].notna().any(axis=1) |
            df["LAI"].notna() |
            df["LST"].notna()
        )
    )
    df["valid_for_inference"] = valid_for_inference.astype(np.uint8)

    meta = {
        "date": pd.Timestamp(date_py),
        "shape": TARGET_SHAPE,
        "rows": rows,
        "cols": cols,
        "lat": lat,
        "lon": lon,
        "transform": (-180.0, 0.1, 0.0, 90.0, 0.0, -0.1),
        "crs": "EPSG:4326",
    }

    gc.collect()
    return df, meta


# =============================================================================
# 供预测脚本使用的列排除工具
# =============================================================================
DEFAULT_NON_FEATURE_COLS = {
    TARGET_COL,
    DATE_COL,
    LAT_COL,
    LON_COL,
    "row",
    "col",
    "predictor_valid",
    "valid_for_inference",
}


def infer_feature_columns_from_dataframe(df):
    """
    若 artifact 中没有 feature_columns.json，就用一个稳妥的回退策略：
    - 去掉明显不是特征的列
    - 保留数值列 + object/category 列
    """
    cols = []
    for c in df.columns:
        if c in DEFAULT_NON_FEATURE_COLS:
            continue
        cols.append(c)
    return cols