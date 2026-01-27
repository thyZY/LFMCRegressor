from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import pandas as pd
import numpy as np


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_col(cfg_cols: Dict[str, Any], key: str) -> Optional[str]:
    v = cfg_cols.get(key)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def standardize_columns(df_raw: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Standardize Globe-LFMC v2.0 table into a normalized schema.

    Key rule:
    - Only trust cfg["columns"][...] if that column name truly exists in df.
      Otherwise fallback to known Globe-LFMC v2.0 official names (and common variants).

    Output columns ensured if possible:
      sorting_id, site_name, lat, lon, date, sampling_datetime, lfmc_pct,
      extra_info_flag, isolated_iforest, species, functional_type,
      protocol, source, site_id, year, doy
      + keep all original columns (no dropping)
    """
    df = df_raw.copy()

    # ---- 0) sanitize column names (critical for hidden spaces / NBSP) ----
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]

    cfg_cols = (cfg.get("columns") or {})

    def _resolve(cfg_key: str, candidates: list[str]) -> str | None:
        """
        Use cfg mapping only if it exists in df.columns, else fallback to candidates.
        """
        c = cfg_cols.get(cfg_key)
        if isinstance(c, str):
            c2 = c.replace("\xa0", " ").strip()
            if c2 in df.columns:
                return c2
        return _pick_first_existing(df, candidates)

    # ---- 1) resolve actual source columns (Globe-LFMC v2.0 official names first) ----
    col_sorting = _resolve("sorting_id", ["Sorting ID", "sorting_id", "SortingID", "ID", "id"])
    col_site    = _resolve("site_name",  ["Site name", "Site Name", "site_name", "site"])
    col_lat     = _resolve("lat",        ["Latitude (WGS84, EPSG:4326)", "Latitude", "lat", "LAT"])
    col_lon     = _resolve("lon",        ["Longitude (WGS84, EPSG:4326)", "Longitude", "lon", "LON"])
    col_date    = _resolve("date",       ["Sampling date (YYYYMMDD)", "Sampling Date (YYYYMMDD)", "Sampling date", "date", "Date"])
    col_time    = _resolve("time",       ["Sampling time (24h format)", "Sampling Time (24h format)", "Sampling time", "time", "Time"])
    col_lfmc    = _resolve("lfmc",       ["LFMC value (%)", "LFMC (%)", "LFMC", "lfmc", "lfmc_pct"])

    col_extra   = _resolve("extra_info_flag",  ["Extra information/Quality Flag", "Extra Info Flag", "extra_info_flag"])
    col_iso     = _resolve("isolated_iforest", ["Isolated data point", "isolated_iforest", "isolated"])
    col_species = _resolve("species",          ["Species collected", "species", "Species"])
    col_ftype   = _resolve("functional_type",  ["Species functional type", "Functional type", "functional_type"])
    col_protocol= _resolve("protocol",         ["Protocol", "protocol"])
    col_source  = _resolve("source",           ["Reference", "Database", "source", "dataset", "Source"])

    # ---- 2) rename map ----
    rename_map = {}
    if col_sorting: rename_map[col_sorting] = "sorting_id"
    if col_site:    rename_map[col_site]    = "site_name"
    if col_lat:     rename_map[col_lat]     = "lat"
    if col_lon:     rename_map[col_lon]     = "lon"
    if col_date:    rename_map[col_date]    = "date"                  # ✅ 你要的：原始采样日期 -> date
    if col_time:    rename_map[col_time]    = "sampling_time_raw"
    if col_lfmc:    rename_map[col_lfmc]    = "lfmc_pct"

    if col_extra:   rename_map[col_extra]   = "extra_info_flag"
    if col_iso:     rename_map[col_iso]     = "isolated_iforest_raw"
    if col_species: rename_map[col_species] = "species"
    if col_ftype:   rename_map[col_ftype]   = "functional_type"
    if col_protocol:rename_map[col_protocol]= "protocol"
    if col_source:  rename_map[col_source]  = "source"

    df = df.rename(columns=rename_map)

    # ---- 3) coerce numeric columns (only if present; do NOT wipe valid data) ----
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "lfmc_pct" in df.columns:
        df["lfmc_pct"] = pd.to_numeric(df["lfmc_pct"], errors="coerce")

    # ---- 4) parse date -> datetime64 (critical: keep as datetime, do NOT cast to str) ----
    if "date" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            dt = pd.to_datetime(df["date"], errors="coerce")
        else:
            s = df["date"]

            # try numeric yyyymmdd
            s_num = pd.to_numeric(s, errors="coerce")
            dt = pd.Series(pd.NaT, index=df.index)

            remain = s_num.notna()
            if remain.any():
                ymd = s_num.loc[remain].astype("Int64").astype(str)
                m8 = ymd.str.fullmatch(r"\d{8}")
                if m8.any():
                    dt.loc[ymd.index[m8]] = pd.to_datetime(ymd[m8], format="%Y%m%d", errors="coerce")

            missing = dt.isna()
            if missing.any():
                dt.loc[missing] = pd.to_datetime(
                    s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan}),
                    errors="coerce"
                )

        df["date"] = dt.dt.floor("D")
    else:
        # 如果 date 列都没匹配到，这里就别强行造一个空 date（否则你会以为“被标准化了但全空”）
        # 直接返回，让你一眼发现配置/列名不匹配
        raise KeyError("Cannot find sampling date column. Check original column name or cfg['columns']['date'].")

    # ---- 5) sampling_datetime = date + time (if time usable) ----
    df["sampling_datetime"] = df["date"]
    if "sampling_time_raw" in df.columns:
        t = df["sampling_time_raw"].astype(str).str.strip()

        # normalize "HHMM" -> "HH:MM"
        t2 = t.where(t.str.contains(":"), t.str.replace(r"^(\d{1,2})(\d{2})$", r"\1:\2", regex=True))
        td = pd.to_timedelta(t2, errors="coerce")
        ok = df["date"].notna() & td.notna()
        df.loc[ok, "sampling_datetime"] = df.loc[ok, "date"] + td.loc[ok]

    # ---- 6) isolated flag ----
    if "isolated_iforest_raw" in df.columns:
        df["isolated_iforest"] = _to_bool(df["isolated_iforest_raw"])
    else:
        df["isolated_iforest"] = False

    # ---- 7) site_id + helper ----
    df["site_id"] = _make_site_id(df)
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear

    return df


def _parse_sampling_datetime(date_series: Optional[pd.Series],
                            time_series: Optional[pd.Series],
                            normalize_to_date: bool = True) -> pd.Series:
    """
    Parse sampling date+time into pandas datetime.
    date_series can be int YYYYMMDD, str, or datetime.
    time_series can be 'hh:mm' (optional).
    """
    if date_series is None:
        return pd.to_datetime(pd.Series([pd.NaT] * (len(time_series) if time_series is not None else 0)))

    ds = date_series.copy()

    # Convert numeric YYYYMMDD -> string
    # Example: 20020601
    if pd.api.types.is_numeric_dtype(ds):
        ds = ds.astype("Int64").astype(str)

    # Strip .0 from excel-like floats converted to strings
    ds = ds.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # If time provided, combine; else parse date only
    if time_series is not None:
        ts = time_series.copy().astype(str).str.strip()
        # some blanks become 'nan'
        ts = ts.replace({"nan": "", "None": ""})
        combined = ds.where(ds != "NaT", "") + " " + ts
        dt = pd.to_datetime(combined, errors="coerce", format=None)
        # fallback: date-only parse
        dt2 = pd.to_datetime(ds, errors="coerce", format="%Y%m%d")
        dt = dt.fillna(dt2)
    else:
        dt = pd.to_datetime(ds, errors="coerce", format="%Y%m%d")

    if normalize_to_date:
        dt = dt.dt.floor("D")

    return dt


def _to_bool(s: pd.Series) -> pd.Series:
    """
    Convert common representations to boolean.
    Accepts: 0/1, '0'/'1', 'true'/'false', 'yes'/'no'
    """
    if s is None:
        return pd.Series(False)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    x = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y"}
    false_set = {"0", "false", "f", "no", "n", "nan", "none", ""}

    out = x.apply(lambda v: True if v in true_set else (False if v in false_set else False))
    return out.astype(bool)


def _make_site_id(df: pd.DataFrame) -> pd.Series:
    """
    Create a stable site_id:
      - Prefer site_name if exists + rounded coords
      - Otherwise use rounded coords only
    """
    site = df["site_name"].astype(str) if "site_name" in df.columns else pd.Series([""] * len(df))
    lat = df["lat"] if "lat" in df.columns else pd.Series([np.nan] * len(df))
    lon = df["lon"] if "lon" in df.columns else pd.Series([np.nan] * len(df))

    lat_r = lat.round(4).astype(str)
    lon_r = lon.round(4).astype(str)

    base = (site.fillna("") + "|" + lat_r + "|" + lon_r).astype(str)

    # hash to keep it compact and stable
    def _h(v: str) -> str:
        return hashlib.md5(v.encode("utf-8")).hexdigest()[:12]

    return base.apply(_h)
