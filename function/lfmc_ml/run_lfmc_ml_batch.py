from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from lfmc_batch_common import (
    OUT_DIR,
    RANDOM_SEED,
    SCHEMES,
    Y_COL,
    build_scheme_frame,
    eval_regression,
    load_base_dataframe,
    save_result_tables,
)


def _encode_category(train_df, val_df, test_df, cat_col):
    all_vals = pd.concat(
        [train_df[cat_col], val_df[cat_col], test_df[cat_col]], axis=0
    ).astype(str).unique()
    mapping = {v: i for i, v in enumerate(all_vals)}

    def _apply(df):
        x = df.copy()
        x[cat_col] = x[cat_col].astype(str).map(mapping).astype(np.int32)
        return x

    return _apply(train_df), _apply(val_df), _apply(test_df)


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


def run_lgbm(train_df, val_df, test_df, num_cols, cat_col, seed):
    train_df, val_df, test_df = _encode_category(train_df, val_df, test_df, cat_col)

    feature_cols = num_cols + [cat_col]
    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    x_test = test_df[feature_cols]

    model = LGBMRegressor(
        objective="mae",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )
    model.fit(
        x_train,
        train_df[Y_COL].to_numpy(),
        sample_weight=train_df["sample_weight"].to_numpy(),
        eval_set=[(x_val, val_df[Y_COL].to_numpy())],
        eval_sample_weight=[val_df["sample_weight"].to_numpy()],
    )

    return {
        "train": model.predict(x_train),
        "val": model.predict(x_val),
        "test": model.predict(x_test),
    }


def run_xgb(train_df, val_df, test_df, num_cols, cat_col, seed):
    train_df, val_df, test_df = _encode_category(train_df, val_df, test_df, cat_col)

    feature_cols = num_cols + [cat_col]
    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    x_test = test_df[feature_cols]

    model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
    )
    model.fit(
        x_train,
        train_df[Y_COL].to_numpy(),
        sample_weight=train_df["sample_weight"].to_numpy(),
        eval_set=[(x_val, val_df[Y_COL].to_numpy())],
        verbose=False,
    )

    return {
        "train": model.predict(x_train),
        "val": model.predict(x_val),
        "test": model.predict(x_test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    base_df, igbp_num_cols = load_base_dataframe()
    rows = []

    for scheme in SCHEMES:
        scheme_df, num_cols, cat_col = build_scheme_frame(base_df, scheme, igbp_num_cols)

        train_df = scheme_df[scheme_df["split"] == "train"].copy()
        val_df = scheme_df[scheme_df["split"] == "val"].copy()
        test_df = scheme_df[scheme_df["split"] == "test"].copy()

        for rep in range(args.repeats):
            seed = RANDOM_SEED + rep

            preds_by_model = {
                "CatBoost": run_catboost(train_df, val_df, test_df, num_cols, cat_col, seed),
                "LightGBM": run_lgbm(train_df, val_df, test_df, num_cols, cat_col, seed),
                "XGBoost": run_xgb(train_df, val_df, test_df, num_cols, cat_col, seed),
            }

            for model_name, preds in preds_by_model.items():
                for split_name, split_df in [
                    ("train", train_df),
                    ("val", val_df),
                    ("test", test_df),
                ]:
                    metrics = eval_regression(split_df[Y_COL].to_numpy(), preds[split_name])
                    rows.append(
                        {
                            "scheme": scheme.code,
                            "scheme_desc": scheme.description,
                            "model": model_name,
                            "repeat": rep + 1,
                            "split": split_name,
                            "n": len(split_df),
                            **metrics,
                        }
                    )

    raw_path, summary_path = save_result_tables(rows, OUT_DIR, "lfmc_7schemes_ml")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
