from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def add_qc_flags(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add QC flags WITHOUT dropping rows.
    Flags:
      - qc_hard: 1 if missing critical fields or impossible values
      - bad_coord: 1 if lat/lon invalid
      - bad_lfmc: 1 if lfmc outside [min,max] or negative
      - bad_date: 1 if date missing
      - site_coord_conflict: 1 if same site_name maps to multiple coords
      - coord_site_conflict: 1 if same coords map to multiple site_name
      - provider_suspect: 1 if extra_info_flag contains keywords
      - isolated_iforest: already standardized boolean (kept)
      - dead_protocol: 1 if protocol code indicates dead/senescent/litter/mixed material (SOFT)
    """
    out = df.copy()

    lfmc_min = float(cfg.get("lfmc_min", 0))
    lfmc_max = float(cfg.get("lfmc_max", 400))
    keywords = [k.lower() for k in (cfg.get("suspect_keywords") or [])]

    # ===================== 0) dead protocol (SOFT flag) =====================
    # Default codes from your protocol sheet interpretation:
    # 7, 9 (live+dead), 14b (senescent), 17 (dead material possible),
    # 27 (dead leaves), 32c (moss & litter)
    default_dead_codes = ["7", "9", "14b", "17", "27", "32c"]
    dead_codes = cfg.get("dead_protocol_codes", default_dead_codes)
    dead_codes = [str(x).strip().lower() for x in (dead_codes or [])]

    out["dead_protocol"] = 0
    if "protocol" in out.columns and len(dead_codes) > 0:
        prot = out["protocol"].astype(str).str.strip().str.lower()
        # handle numeric protocols read as "7.0"
        prot = prot.str.replace(r"\.0$", "", regex=True)
        out.loc[prot.isin(set(dead_codes)), "dead_protocol"] = 1
    out["dead_protocol"] = out["dead_protocol"].astype(np.uint8)

    # ----- basic availability -----
    out["bad_date"] = out["date"].isna().astype(np.uint8)
    out["bad_coord"] = (
        out["lat"].isna() | out["lon"].isna() |
        (out["lat"] < -90) | (out["lat"] > 90) |
        (out["lon"] < -180) | (out["lon"] > 180)
    ).astype(np.uint8)

    out["bad_lfmc"] = (
        out["lfmc_pct"].isna() |
        (out["lfmc_pct"] < lfmc_min) |
        (out["lfmc_pct"] > lfmc_max)
    ).astype(np.uint8)

    # hard QC: any of these means not usable as numeric training label
    out["qc_hard"] = ((out["bad_date"] == 1) | (out["bad_coord"] == 1) | (out["bad_lfmc"] == 1)).astype(np.uint8)

    # ----- provider suspect flag from extra_info_flag text -----
    out["provider_suspect"] = 0
    if "extra_info_flag" in out.columns and len(keywords) > 0:
        txt = out["extra_info_flag"].fillna("").astype(str).str.lower()
        mask = np.zeros(len(out), dtype=bool)
        for kw in keywords:
            mask |= txt.str.contains(kw, na=False)
        out.loc[mask, "provider_suspect"] = 1
    out["provider_suspect"] = out["provider_suspect"].astype(np.uint8)

    # ----- site <-> coord consistency checks (soft flags) -----
    out["site_coord_conflict"] = 0
    out["coord_site_conflict"] = 0

    if "site_name" in out.columns:
        # site_name -> number of unique coords
        tmp = out.loc[(out["bad_coord"] == 0), ["site_name", "lat", "lon"]].copy()
        tmp["site_name"] = tmp["site_name"].astype(str)
        g1 = tmp.groupby("site_name")[["lat", "lon"]].apply(lambda x: x.drop_duplicates().shape[0])
        conflict_sites = set(g1[g1 > 1].index.tolist())
        out.loc[out["site_name"].astype(str).isin(conflict_sites), "site_coord_conflict"] = 1

        # coords -> number of unique site_name
        tmp2 = tmp.copy()
        tmp2["lat_r"] = tmp2["lat"].round(4)
        tmp2["lon_r"] = tmp2["lon"].round(4)
        g2 = tmp2.groupby(["lat_r", "lon_r"])["site_name"].nunique()
        conflict_coords = set(g2[g2 > 1].index.tolist())
        coord_key = list(zip(out["lat"].round(4), out["lon"].round(4)))
        out.loc[pd.Index(coord_key).isin(conflict_coords), "coord_site_conflict"] = 1

    out["site_coord_conflict"] = out["site_coord_conflict"].astype(np.uint8)
    out["coord_site_conflict"] = out["coord_site_conflict"].astype(np.uint8)

    # ----- report -----
    report = {
        "n_rows": int(len(out)),
        "n_qc_hard": int(out["qc_hard"].sum()),
        "n_bad_date": int(out["bad_date"].sum()),
        "n_bad_coord": int(out["bad_coord"].sum()),
        "n_bad_lfmc": int(out["bad_lfmc"].sum()),
        "n_provider_suspect": int(out["provider_suspect"].sum()),
        "n_isolated_iforest": int(out["isolated_iforest"].sum()) if "isolated_iforest" in out.columns else None,
        "n_site_coord_conflict": int(out["site_coord_conflict"].sum()),
        "n_coord_site_conflict": int(out["coord_site_conflict"].sum()),
        "n_dead_protocol": int(out["dead_protocol"].sum()),
    }

    return out, report


def make_strict_view(
    df: pd.DataFrame,
    drop_isolated_iforest: bool = False,
    drop_provider_suspect: bool = False,
    drop_dead_protocol: bool = False,
) -> pd.DataFrame:
    """
    Build a strict view for modeling (still without VOD matching).

    Views are controlled by flags:
      - Always drops qc_hard==1 by default.
      - Optionally drops isolated_iforest/provider_suspect/dead_protocol.

    Note:
      - dead_protocol is a SOFT flag; only dropped if drop_dead_protocol=True.
    """
    m = (df["qc_hard"] == 0)

    if drop_isolated_iforest and "isolated_iforest" in df.columns:
        m &= (df["isolated_iforest"] == False)

    if drop_provider_suspect and "provider_suspect" in df.columns:
        m &= (df["provider_suspect"] == 0)

    if drop_dead_protocol and "dead_protocol" in df.columns:
        m &= (df["dead_protocol"] == 0)

    return df.loc[m].copy()


def make_hard_only_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only qc_hard==1 rows for debugging/data auditing.
    """
    if "qc_hard" not in df.columns:
        raise KeyError("qc_hard column not found. Run add_qc_flags first.")
    return df.loc[df["qc_hard"] == 1].copy()

