from __future__ import annotations

import argparse

import torch

from lfmc_final_common import (
    OUT_DIR,
    RANDOM_SEED,
    build_final_s6_lite_frame,
    load_base_dataframe,
    save_result_tables,
)
from lfmc_final_dl_moe_common import (
    MoEMLPRegressor,
    MoEResNetMLPRegressor,
    fit_model,
    make_dataloaders,
    prepare_dl_category,
    prepare_dl_numeric_features,
    run_epoch,
    seed_everything,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_df = load_base_dataframe()
    final_df, num_cols, cat_col = build_final_s6_lite_frame(base_df)

    train_df = final_df[final_df["split"] == "train"].copy()
    val_df = final_df[final_df["split"] == "val"].copy()
    test_df = final_df[final_df["split"] == "test"].copy()

    train_df, val_df, test_df, n_cats = prepare_dl_category(train_df, val_df, test_df, cat_col)
    train_df, val_df, test_df = prepare_dl_numeric_features(train_df, val_df, test_df, num_cols)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_df, val_df, test_df, num_cols, cat_col
    )

    num_dim = len(num_cols)

    print("=== Final S6-lite DL MoE dataset ===")
    print(f"rows: {len(final_df)}")
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"num features: {len(num_cols) + 1} (numeric={len(num_cols)}, cat=1)")
    print("numeric columns:", num_cols)
    print("categorical column:", cat_col)
    print("n_cats:", n_cats)

    rows = []

    for rep in range(args.repeats):
        seed = RANDOM_SEED + rep
        seed_everything(seed)

        models = {
            "MoE_MLP": MoEMLPRegressor(
                num_dim=num_dim,
                n_cats=n_cats,
                emb_dim=16,
                n_experts=4,
                expert_hidden=(256, 128),
                gate_hidden=128,
                dropout=0.2,
            ),
            "MoE_ResNetMLP": MoEResNetMLPRegressor(
                num_dim=num_dim,
                n_cats=n_cats,
                emb_dim=16,
                n_experts=4,
                width=256,
                n_blocks=3,
                gate_hidden=128,
                dropout=0.2,
            ),
        }

        for model_name, model in models.items():
            print(f"[repeat {rep + 1}/{args.repeats}] training {model_name}")
            trained = fit_model(model, train_loader, val_loader, device)

            _, train_metrics = run_epoch(trained, train_loader, device)
            _, val_metrics = run_epoch(trained, val_loader, device)
            _, test_metrics = run_epoch(trained, test_loader, device)

            for split_name, split_df, metrics in [
                ("train", train_df, train_metrics),
                ("val", val_df, val_metrics),
                ("test", test_df, test_metrics),
            ]:
                rows.append(
                    {
                        "scheme": "FINAL_S6_LITE",
                        "scheme_desc": "lc_dom + VOD + LAI/Hveg/LST + doy_sin/doy_cos/lat_norm + consistency",
                        "model": model_name,
                        "repeat": rep + 1,
                        "split": split_name,
                        "n": len(split_df),
                        **metrics,
                    }
                )

    raw_path, summary_path = save_result_tables(rows, OUT_DIR, "lfmc_final_s6_lite_dl_moe")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
