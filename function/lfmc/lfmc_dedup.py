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


def deduplicate_site_day(
    df: pd.DataFrame,
    strategy: Literal["median", "mean", "last"] = "median",
    group_by_vegtype: bool = True,
    veg_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate LFMC observations per site per day, optionally split by vegetation type.

    Required columns:
      - site_id, date, lfmc_pct, lat, lon
    If group_by_vegtype=True, uses veg_col (auto-detected if None).
      Recommended veg_col: 'species_functional_type' (or your standardized alias).

    Output columns:
      - site_id, date, veg_type (if enabled), lat, lon
      - lfmc_pct (aggregated)
      - n_obs (group size)
      - veg_type_unknown_flag (1 if veg_type == 'Unknown')
      - plus optional descriptor cols: site_name/source/protocol/species/individual_or_mean (first non-null)
    """
    required = {"site_id", "date", "lfmc_pct", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"deduplicate_site_day missing required columns: {sorted(missing)}")

    work = df.copy()

    # ---- choose vegetation functional type column ----
    if group_by_vegtype:
        if veg_col is None:
            # try common standardized names (depending on your standardize step)
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

    # ---- group keys ----
    group_keys = ["site_id", "date", "veg_type"] if group_by_vegtype else ["site_id", "date"]

    # stable ordering (supports 'last' strategy)
    sort_keys = group_keys.copy()
    if "sampling_datetime" in work.columns:
        sort_keys.append("sampling_datetime")
    work = work.sort_values(sort_keys)

    g = work.groupby(group_keys, as_index=False)

    # ---- aggregate LFMC ----
    if strategy == "median":
        agg_lfmc = g["lfmc_pct"].median()
    elif strategy == "mean":
        agg_lfmc = g["lfmc_pct"].mean()
    elif strategy == "last":
        agg_lfmc = g["lfmc_pct"].last()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ---- representative lat/lon ----
    agg_lat = g["lat"].median()
    agg_lon = g["lon"].median()

    # ---- group size ----
    n_obs = g.size().rename(columns={"size": "n_obs"})

    # ---- keep some optional descriptor cols (first non-null) ----
    def first_nonnull(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[0] if len(s2) else None

    keep_cols = []
    for c in ["site_name", "source", "protocol", "species", "individual_or_mean"]:
        if c in work.columns:
            keep_cols.append(c)

    out = (
        agg_lfmc
        .merge(agg_lat, on=group_keys, how="left")
        .merge(agg_lon, on=group_keys, how="left")
        .merge(n_obs, on=group_keys, how="left")
    )

    if keep_cols:
        extra = g[keep_cols].agg(first_nonnull)
        out = out.merge(extra, on=group_keys, how="left")

    # ---- Unknown flag ----
    if group_by_vegtype:
        out["veg_type_unknown_flag"] = (out["veg_type"] == "Unknown").astype(np.uint8)
    else:
        out["veg_type_unknown_flag"] = 0

    return out
