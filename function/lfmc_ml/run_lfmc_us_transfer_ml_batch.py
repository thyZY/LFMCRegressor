from __future__ import annotations

import argparse

import numpy as np
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from lfmc_final_common import Y_COL
from lfmc_us_transfer_common import (
    RANDOM_SEED,
    SPLIT_ORDER,
    build_us_transfer_s6_lite_frame,
    collect_us_transfer_rows,
    print_us_transfer_dataset_summary,
    print_us_transfer_metrics,
    save_result_tables,
    split_frame_by_eval_group,
)


def _encode_category(split_frames, cat_col):
    all_vals = np.concatenate(
        [split_frames[name][cat_col].astype(str).to_numpy() for name in SPLIT_ORDER if len(split_frames[name]) > 0]
    )
    uniq = np.unique(all_vals)
    mapping = {v: i for i, v in enumerate(uniq)}

    out = {}
    for split_name, df in split_frames.items():
        x = df.copy()
        x[cat_col] = x[cat_col].astype(str).map(mapping).astype(np.int32)
        out[split_name] = x
    return out


def run_catboost(split_frames, num_cols, cat_col, seed):
    feature_cols = num_cols + [cat_col]

    train_pool = Pool(
        split_frames["train_us"][feature_cols],
        split_frames["train_us"][Y_COL],
        cat_features=[cat_col],
        weight=split_frames["train_us"]["sample_weight"],
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
    val_pool = Pool(
        split_frames["val_us_overlap"][feature_cols],
        split_frames["val_us_overlap"][Y_COL],
        cat_features=[cat_col],
        weight=split_frames["val_us_overlap"]["sample_weight"],
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = {}
    for split_name, df in split_frames.items():
        pool = Pool(df[feature_cols], df[Y_COL], cat_features=[cat_col], weight=df["sample_weight"])
        preds[split_name] = model.predict(pool)
    return preds


def run_lgbm(split_frames, num_cols, cat_col, seed):
    encoded = _encode_category(split_frames, cat_col)
    feature_cols = num_cols + [cat_col]

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
        encoded["train_us"][feature_cols],
        encoded["train_us"][Y_COL].to_numpy(),
        sample_weight=encoded["train_us"]["sample_weight"].to_numpy(),
        eval_set=[(encoded["val_us_overlap"][feature_cols], encoded["val_us_overlap"][Y_COL].to_numpy())],
        eval_sample_weight=[encoded["val_us_overlap"]["sample_weight"].to_numpy()],
    )

    return {split_name: model.predict(df[feature_cols]) for split_name, df in encoded.items()}


def run_xgb(split_frames, num_cols, cat_col, seed):
    encoded = _encode_category(split_frames, cat_col)
    feature_cols = num_cols + [cat_col]

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
        encoded["train_us"][feature_cols],
        encoded["train_us"][Y_COL].to_numpy(),
        sample_weight=encoded["train_us"]["sample_weight"].to_numpy(),
        eval_set=[(encoded["val_us_overlap"][feature_cols], encoded["val_us_overlap"][Y_COL].to_numpy())],
        verbose=False,
    )

    return {split_name: model.predict(df[feature_cols]) for split_name, df in encoded.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=r"d:\Python\jupyter\jupyter\LFMCRegressor\figs\us_transfer_runs")
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()
    from pathlib import Path
    out_dir = Path(args.out_dir)

    final_df, num_cols, cat_col = build_us_transfer_s6_lite_frame(args.data_path)
    print_us_transfer_dataset_summary(final_df, num_cols, cat_col)
    split_frames = split_frame_by_eval_group(final_df)

    rows = []
    for rep in range(args.repeats):
        seed = RANDOM_SEED + rep
        preds_by_model = {
            "CatBoost": run_catboost(split_frames, num_cols, cat_col, seed),
            "LightGBM": run_lgbm(split_frames, num_cols, cat_col, seed),
            "XGBoost": run_xgb(split_frames, num_cols, cat_col, seed),
        }

        for model_name, preds in preds_by_model.items():
            metrics_by_split = collect_us_transfer_rows(
                rows=rows,
                model_name=model_name,
                repeat_idx=rep + 1,
                split_frames=split_frames,
                preds_by_split=preds,
                scheme="US_TRANSFER_S6_LITE",
                scheme_desc="US train/val + external world test on S6-lite",
            )
            print_us_transfer_metrics(model_name, rep + 1, metrics_by_split)

    raw_path, summary_path = save_result_tables(rows, out_dir, "lfmc_us_transfer_ml")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
