from __future__ import annotations

from typing import Literal, Optional
import pandas as pd
import numpy as np


def _normalize_veg_type(s: pd.Series) -> pd.Series:
    """
    Map 'Species functional type' into {Grass, Shrub, Tree, Unknown}.

    Based on the value list you showed (partial):
      Grass-like: Grass, Graminoid, Sedge, Forb, Geophyte
      Shrub-like: Shrub, Large shrub, Subshrub
      Tree-like:  Tree, Small tree
      Borderline/special: Shrub/Small tree, Liana, Moss and litter, Obligate hemiparasite -> Unknown

    Notes:
    - We intentionally keep Unknown (do NOT force into grass/shrub/tree),
      to avoid structural bias. You can later choose to keep/drop Unknown.
    """
    x = s.fillna("").astype(str).str.strip()

    def map_one(v: str) -> str:
        v_low = v.lower()

        # ---- grass / herbaceous ----
        if v_low in {"grass", "graminoid", "sedge", "forb", "geophyte"}:
            return "Grass"

        # ---- shrub / low woody ----
        if v_low in {"shrub", "large shrub", "subshrub"}:
            return "Shrub"

        # ---- tree / woody canopy ----
        if v_low in {"tree", "small tree"}:
            return "Tree"

        # ---- borderline / special classes -> Unknown ----
        if v_low in {"shrub/small tree", "liana", "moss and litter", "obligate hemiparasite"}:
            return "Unknown"

        # fallback for unseen values
        return "Unknown"

    return x.apply(map_one)

def _mode_or_none(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return None
    m = s2.mode()
    return m.iloc[0] if len(m) else None


def _build_site_loc_id(lat: pd.Series, lon: pd.Series, precision_deg: float) -> pd.Series:
    """
    Quantize lat/lon to a fixed grid, then build a stable location id.
    precision_deg:
      - 1e-4 ~ 11 m (equator)
      - 1e-3 ~ 111 m
    """
    lat_q = (lat.astype(float) / precision_deg).round().astype("Int64")
    lon_q = (lon.astype(float) / precision_deg).round().astype("Int64")
    return lat_q.astype(str) + "_" + lon_q.astype(str)


def deduplicate_site_day(
    df: pd.DataFrame,
    strategy: Literal["median", "mean", "last"] = "median",
    group_by_vegtype: bool = True,
    veg_col: Optional[str] = None,
    # NEW: group by quantized lat/lon instead of site_id
    group_by_location: bool = False,
    location_precision_deg: float = 1e-4,
    location_id_col: str = "site_loc_id",
    # NEW: keep IGBP land cover (mode + nunique)
    keep_igbp: bool = False,
    igbp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate LFMC observations per (site/day), optionally split by vegetation type.
    Can optionally use quantized (lat, lon) as the grouping "site" (group_by_location=True),
    but WILL still keep a representative site_id in outputs for traceability.

    Required columns (always):
      - date, lfmc_pct, lat, lon
    Additionally required when group_by_location=False:
      - site_id

    If group_by_vegtype=True, uses veg_col (auto-detected if None).
      Recommended veg_col: 'species_functional_type' (or your standardized alias).

    Output columns:
      - date, lat, lon
      - site_id (always kept; representative if group_by_location=True)
      - site_loc_id (only when group_by_location=True)
      - veg_type (if enabled)
      - lfmc_pct (aggregated)
      - n_obs (group size)
      - veg_type_unknown_flag (1 if veg_type == 'Unknown')
      - site_id_nunique (only when group_by_location=True; QC)
      - optional: igbp_500m, igbp_500m_nunique (if keep_igbp=True and column exists)
      - plus optional descriptor cols: site_name/source/protocol/species/individual_or_mean (first non-null)
    """
    # ---- required columns ----
    base_required = {"date", "lfmc_pct", "lat", "lon"}
    if not group_by_location:
        base_required.add("site_id")

    missing = base_required - set(df.columns)
    if missing:
        raise ValueError(f"deduplicate_site_day missing required columns: {sorted(missing)}")

    work = df.copy()

    # ---- drop invalid rows to avoid fake "<NA>_<NA>" locations or NaT groups ----
    work = work.dropna(subset=["date", "lfmc_pct", "lat", "lon"])

    # ---- choose vegetation functional type column ----
    if group_by_vegtype:
        if veg_col is None:
            for c in ["species_functional_type", "functional_type", "Species functional type"]:
                if c in work.columns:
                    veg_col = c
                    break
        if veg_col is None:
            work["veg_type"] = "Unknown"
        else:
            work["veg_type"] = _normalize_veg_type(work[veg_col])
    else:
        work["veg_type"] = "All"

    # ---- IGBP column (optional) ----
    if keep_igbp:
        if igbp_col is None:
            for c in ["IGBP Land Cover", "igbp_500m", "igbp"]:
                if c in work.columns:
                    igbp_col = c
                    break
        if igbp_col is None:
            keep_igbp = False  # silently disable if not found

    # ---- helper: first non-null ----
    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else None

    # ---- helper: representative site_id (prefer mode; fallback first) ----
    def site_id_rep(s: pd.Series):
        s2 = s.dropna()
        if len(s2) == 0:
            return None
        m = s2.mode()
        return m.iloc[0] if len(m) else s2.iloc[0]

    # ---- group keys ----
    if group_by_location:
        work[location_id_col] = _build_site_loc_id(work["lat"], work["lon"], location_precision_deg)
        group_keys = [location_id_col, "date", "veg_type"] if group_by_vegtype else [location_id_col, "date"]
    else:
        group_keys = ["site_id", "date", "veg_type"] if group_by_vegtype else ["site_id", "date"]

    # stable ordering (supports 'last' strategy)
    sort_keys = group_keys.copy()
    if "sampling_datetime" in work.columns:
        sort_keys.append("sampling_datetime")
    work = work.sort_values(sort_keys)

    g = work.groupby(group_keys, as_index=False)

    # ---- aggregate LFMC ----
    strategy_l = (strategy or "median").lower()
    if strategy_l == "median":
        agg_lfmc = g["lfmc_pct"].median()
    elif strategy_l == "mean":
        agg_lfmc = g["lfmc_pct"].mean()
    elif strategy_l == "last":
        agg_lfmc = g["lfmc_pct"].last()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ---- representative lat/lon ----
    agg_lat = g["lat"].median()
    agg_lon = g["lon"].median()

    # ---- group size ----
    n_obs = g.size().rename(columns={"size": "n_obs"})

    out = (
        agg_lfmc
        .merge(agg_lat, on=group_keys, how="left")
        .merge(agg_lon, on=group_keys, how="left")
        .merge(n_obs, on=group_keys, how="left")
    )

    # ---- keep optional descriptor cols (first non-null) ----
    keep_cols = []
    for c in ["site_name", "source", "protocol", "species", "individual_or_mean"]:
        if c in work.columns:
            keep_cols.append(c)

    if keep_cols:
        extra = g[keep_cols].agg(first_nonnull)
        out = out.merge(extra, on=group_keys, how="left")

    # ---- keep IGBP (mode + nunique) ----
    if keep_igbp and igbp_col is not None:
        igbp_stats = g[igbp_col].agg(
            igbp_500m=_mode_or_none,
            igbp_500m_nunique=lambda x: x.dropna().nunique(),
        )
        out = out.merge(igbp_stats, on=group_keys, how="left")

    # ---- ALWAYS keep site_id in outputs ----
    if "site_id" in work.columns:
        if group_by_location:
            # many site_id can map to same location-day-veg_type; keep representative + QC
            sid_rep = g["site_id"].agg(site_id=site_id_rep)
            sid_nuniq = g["site_id"].agg(site_id_nunique=lambda x: x.dropna().nunique())
            out = out.merge(sid_rep, on=group_keys, how="left").merge(sid_nuniq, on=group_keys, how="left")
        else:
            # site_id is part of group_keys already; nothing extra needed
            pass
    else:
        # if caller truly has no site_id column, still create it for schema consistency
        out["site_id"] = None

    # ---- Unknown flag ----
    if group_by_vegtype:
        out["veg_type_unknown_flag"] = (out["veg_type"] == "Unknown").astype(np.uint8)
    else:
        out["veg_type_unknown_flag"] = 0

    return out
