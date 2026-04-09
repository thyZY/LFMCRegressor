from __future__ import annotations

import argparse

from catboost import CatBoostRegressor, Pool

from lfmc_ablation_common import (
    ABLATIONS,
    OUT_DIR,
    RANDOM_SEED,
    Y_COL,
    build_ablation_frame,
    eval_regression,
    load_base_dataframe,
    save_result_tables,
)


def run_catboost(train_df, val_df, test_df, num_cols, cat_col, seed):
    feature_cols = num_cols + [cat_col]

    train_pool = Pool(
        train_df[feature_cols],
        train_df[Y_COL],
        cat_features=[cat_col],
        weight=train_df["sample_weight"],
    )
    val_pool = Pool(
        val_df[feature_cols],
        val_df[Y_COL],
        cat_features=[cat_col],
        weight=val_df["sample_weight"],
    )
    test_pool = Pool(
        test_df[feature_cols],
        test_df[Y_COL],
        cat_features=[cat_col],
        weight=test_df["sample_weight"],
    )

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        random_seed=seed,
        od_type="Iter",
        od_wait=150,
        verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    return {
        "train": model.predict(train_pool),
        "val": model.predict(val_pool),
        "test": model.predict(test_pool),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    base_df, igbp_num_cols = load_base_dataframe()
    rows = []

    for spec in ABLATIONS:
        print(f"\n=== Running {spec.code} | {spec.description} ===")

        ab_df, num_cols, cat_col = build_ablation_frame(base_df, spec, igbp_num_cols)

        train_df = ab_df[ab_df["split"] == "train"].copy()
        val_df = ab_df[ab_df["split"] == "val"].copy()
        test_df = ab_df[ab_df["split"] == "test"].copy()

        print(f"rows: {len(ab_df)}")
        print(
            f"split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        print(
            f"weight min/max: {train_df['sample_weight'].min():.4f}, "
            f"{train_df['sample_weight'].max():.4f}"
        )
        print(f"num features: {len(num_cols) + 1} (numeric={len(num_cols)}, cat=1)")

        for rep in range(args.repeats):
            seed = RANDOM_SEED + rep
            print(f"  [{spec.code}] CatBoost repeat {rep + 1}/{args.repeats}")

            preds = run_catboost(train_df, val_df, test_df, num_cols, cat_col, seed)

            for split_name, split_df in [
                ("train", train_df),
                ("val", val_df),
                ("test", test_df),
            ]:
                metrics = eval_regression(split_df[Y_COL].to_numpy(), preds[split_name])
                rows.append(
                    {
                        "base_scheme": "S6_LC_DOM_FE_W_CONS",
                        "ablation": spec.code,
                        "ablation_desc": spec.description,
                        "model": "CatBoost",
                        "repeat": rep + 1,
                        "split": split_name,
                        "n": len(split_df),
                        **metrics,
                    }
                )

    raw_path, summary_path = save_result_tables(rows, OUT_DIR, "lfmc_s6_catboost_ablation")
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
