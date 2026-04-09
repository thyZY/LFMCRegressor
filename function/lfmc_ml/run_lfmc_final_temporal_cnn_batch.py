from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lfmc_final_common import (
    RANDOM_SEED,
    build_final_s6_lite_frame,
    load_base_dataframe,
    save_result_tables,
)
from lfmc_final_temporal_cnn_common import (
    CAT_COL_DEFAULT,
    DYNAMIC_COLS_DEFAULT,
    STATIC_NUM_COLS_DEFAULT,
    DualBranchTemporalCNNRegressor,
    build_temporal_windows,
    fit_model,
    make_temporal_dataloaders,
    prepare_temporal_category,
    prepare_temporal_numeric_features,
    run_epoch,
    save_temporal_cnn_artifact,
    seed_everything,
)


DEFAULT_OUT_DIR = Path(r"d:\Python\jupyter\jupyter\LFMCRegressor\notebooks\artifacts\dl_temporal")


def print_split_metrics(model_name: str, repeat_idx: int, metrics_by_split: dict[str, dict[str, float]]):
    print(f"[repeat {repeat_idx}] {model_name} metrics")
    for split_name in ["train", "val", "test"]:
        m = metrics_by_split[split_name]
        print(
            f"  {split_name:<5} "
            f"MAE={m['MAE']:.4f}  "
            f"RMSE={m['RMSE']:.4f}  "
            f"R={m['R']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--window-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)

    base_df = load_base_dataframe(Path(args.data_path) if args.data_path else None)
    final_df, _, _ = build_final_s6_lite_frame(base_df)

    temporal_df = build_temporal_windows(
        final_df,
        window_size=args.window_size,
        static_num_cols=STATIC_NUM_COLS_DEFAULT,
        dynamic_cols=DYNAMIC_COLS_DEFAULT,
        cat_col=CAT_COL_DEFAULT,
    )
    if temporal_df.empty:
        raise ValueError("No temporal samples were created. Try a smaller --window-size.")

    train_base = temporal_df[temporal_df["split"] == "train"].copy()
    val_base = temporal_df[temporal_df["split"] == "val"].copy()
    test_base = temporal_df[temporal_df["split"] == "test"].copy()

    print("Device:", device)
    print("=== Final S6-lite Dual-Branch Temporal CNN dataset ===")
    print(f"window_size: {args.window_size}")
    print(f"rows: {len(temporal_df)}")
    print(f"train={len(train_base)}, val={len(val_base)}, test={len(test_base)}")
    print("static numeric columns:", STATIC_NUM_COLS_DEFAULT)
    print("dynamic columns:", DYNAMIC_COLS_DEFAULT)
    print("categorical column:", CAT_COL_DEFAULT)

    rows = []
    for rep in range(args.repeats):
        seed_everything(RANDOM_SEED + rep)

        train_df = train_base.copy()
        val_df = val_base.copy()
        test_df = test_base.copy()

        train_df, val_df, test_df, n_cats, encoder = prepare_temporal_category(
            train_df, val_df, test_df, CAT_COL_DEFAULT
        )
        train_df, val_df, test_df, stats = prepare_temporal_numeric_features(
            train_df, val_df, test_df, STATIC_NUM_COLS_DEFAULT, DYNAMIC_COLS_DEFAULT
        )

        train_loader, val_loader, test_loader = make_temporal_dataloaders(
            train_df, val_df, test_df, STATIC_NUM_COLS_DEFAULT, CAT_COL_DEFAULT
        )

        model_kwargs = {
            "static_dim": len(STATIC_NUM_COLS_DEFAULT),
            "dynamic_dim": len(DYNAMIC_COLS_DEFAULT),
            "n_cats": n_cats,
            "emb_dim": 8,
            "static_hidden": 64,
            "temporal_channels": 64,
            "fusion_hidden": 128,
            "dropout": 0.2,
        }
        model = DualBranchTemporalCNNRegressor(**model_kwargs)
        print(f"[repeat {rep + 1}/{args.repeats}] training DualBranchTemporalCNN")
        trained = fit_model(model, train_loader, val_loader, device)

        artifact_dir = out_dir / f"DualBranchTemporalCNN_rep{rep + 1}"
        save_temporal_cnn_artifact(
            artifact_dir=artifact_dir,
            model=trained,
            model_name="DualBranchTemporalCNN",
            model_kwargs=model_kwargs,
            window_size=args.window_size,
            static_num_cols=STATIC_NUM_COLS_DEFAULT,
            dynamic_cols=DYNAMIC_COLS_DEFAULT,
            cat_col=CAT_COL_DEFAULT,
            stats=stats,
            encoder=encoder,
        )

        _, train_metrics = run_epoch(trained, train_loader, device)
        _, val_metrics = run_epoch(trained, val_loader, device)
        _, test_metrics = run_epoch(trained, test_loader, device)
        print_split_metrics(
            "DualBranchTemporalCNN",
            rep + 1,
            {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
        )

        for split_name, split_df, metrics in [
            ("train", train_df, train_metrics),
            ("val", val_df, val_metrics),
            ("test", test_df, test_metrics),
        ]:
            rows.append(
                {
                    "scheme": "FINAL_S6_LITE_TEMPORAL",
                    "scheme_desc": "dual-branch temporal cnn on S6-lite windows",
                    "model": "DualBranchTemporalCNN",
                    "repeat": rep + 1,
                    "split": split_name,
                    "n": len(split_df),
                    "window_size": args.window_size,
                    **metrics,
                }
            )

    raw_path, summary_path = save_result_tables(rows, out_dir, "lfmc_final_s6_lite_temporal_cnn")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
