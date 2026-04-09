from __future__ import annotations

import pandas as pd

from lfmc_ablation_v2_common import (
    FT_COL,
    IGBP_COL,
    LC_NONVEG_COLS,
    NONVEG_THRESHOLD,
    OUT_DIR,
    Y_COL,
    build_ablation_v2_frame,
    load_base_dataframe,
    spatial_block_split,
    AblationSpecV2,
)


def main():
    base_df, igbp_num_cols = load_base_dataframe()

    # FULL_S6 before split-condition filtering view
    full_df = base_df.copy()
    full_df = spatial_block_split(full_df, seed=42)

    full_df["lc_dom_str"] = full_df["lc_dom"].astype("string")
    full_df["is_consistent"] = full_df["igbp_to_lc_dom"] == full_df["lc_dom_str"]

    summary_rows = []

    for split_name in ["train", "val", "test"]:
        sub = full_df[full_df["split"] == split_name].copy()
        consistent = sub[sub["is_consistent"]].copy()
        inconsistent = sub[~sub["is_consistent"]].copy()

        summary_rows.append(
            {
                "split": split_name,
                "n_total": len(sub),
                "n_consistent": len(consistent),
                "n_inconsistent": len(inconsistent),
                "consistent_ratio": len(consistent) / len(sub) if len(sub) else 0.0,
                "inconsistent_ratio": len(inconsistent) / len(sub) if len(sub) else 0.0,
                "lfmc_mean_total": sub[Y_COL].mean(),
                "lfmc_mean_consistent": consistent[Y_COL].mean(),
                "lfmc_mean_inconsistent": inconsistent[Y_COL].mean(),
                "lfmc_std_total": sub[Y_COL].std(),
                "lfmc_std_consistent": consistent[Y_COL].std(),
                "lfmc_std_inconsistent": inconsistent[Y_COL].std(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    by_class_rows = []
    for split_name in ["train", "val", "test"]:
        sub = full_df[full_df["split"] == split_name].copy()

        grp = (
            sub.groupby([IGBP_COL, "lc_dom_str", "is_consistent"], as_index=False)
            .agg(
                n=(Y_COL, "size"),
                lfmc_mean=(Y_COL, "mean"),
                lfmc_std=(Y_COL, "std"),
            )
            .sort_values("n", ascending=False)
        )
        grp["split"] = split_name
        by_class_rows.append(grp)

    by_class_df = pd.concat(by_class_rows, ignore_index=True)

    ft_rows = []
    for split_name in ["train", "val", "test"]:
        sub = full_df[full_df["split"] == split_name].copy()

        grp = (
            sub.groupby([FT_COL, "is_consistent"], as_index=False)
            .agg(
                n=(Y_COL, "size"),
                lfmc_mean=(Y_COL, "mean"),
                lfmc_std=(Y_COL, "std"),
            )
            .sort_values("n", ascending=False)
        )
        grp["split"] = split_name
        ft_rows.append(grp)

    ft_df = pd.concat(ft_rows, ignore_index=True)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "lfmc_s6_consistency_diagnose_summary.csv"
    by_class_path = out_dir / "lfmc_s6_consistency_diagnose_by_class.csv"
    ft_path = out_dir / "lfmc_s6_consistency_diagnose_by_ft.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    by_class_df.to_csv(by_class_path, index=False, encoding="utf-8-sig")
    ft_df.to_csv(ft_path, index=False, encoding="utf-8-sig")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved class details to: {by_class_path}")
    print(f"Saved FT details to: {ft_path}")


if __name__ == "__main__":
    main()
