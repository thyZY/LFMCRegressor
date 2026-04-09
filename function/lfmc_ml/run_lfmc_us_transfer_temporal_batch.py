from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lfmc_us_transfer_common import (
    RANDOM_SEED,
    SPLIT_ORDER,
    build_us_transfer_s6_lite_frame,
    print_us_transfer_dataset_summary,
    print_us_transfer_metrics,
    save_result_tables,
)
from lfmc_final_temporal_cnn_common import (
    CAT_COL_DEFAULT,
    DYNAMIC_COLS_DEFAULT,
    STATIC_NUM_COLS_DEFAULT,
    DualBranchTemporalCNNMoERegressor,
    DualBranchTemporalCNNRegressor,
    DualBranchTemporalCNNResNetMoERegressor,
    DualBranchTemporalCNNResNetRegressor,
    DualBranchTemporalCNNTabNetMoERegressor,
    DualBranchTemporalCNNTabNetRegressor,
    build_temporal_windows,
    fit_model,
    make_temporal_dataloaders,
    prepare_temporal_category,
    prepare_temporal_numeric_features,
    run_epoch,
    save_temporal_cnn_artifact,
    seed_everything,
    transform_temporal_numeric_features,
)


def print_temporal_dataset_summary(temporal_df, window_size):
    print("=== US Transfer Temporal dataset ===")
    print(f"window_size: {window_size}")
    print(f"rows: {len(temporal_df)}")
    for split_name in SPLIT_ORDER:
        sub = temporal_df[temporal_df["split"] == split_name]
        print(f"{split_name}: n={len(sub)}, sites={sub[[ 'Latitude (WGS84, EPSG:4326)', 'Longitude (WGS84, EPSG:4326)' ]].drop_duplicates().shape[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=r"d:\Python\jupyter\jupyter\LFMCRegressor\notebooks\artifacts\dl_us_transfer_temporal")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default="TemporalCNN_ResNetStatic,TemporalCNN_ResNetStatic_MoE,TemporalCNN_TabNetStatic,TemporalCNN_TabNetStatic_MoE",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_df, num_cols, cat_col = build_us_transfer_s6_lite_frame(args.data_path)
    print("Device:", device)
    print_us_transfer_dataset_summary(final_df, num_cols, cat_col)

    temporal_df = build_temporal_windows(
        final_df,
        window_size=args.window_size,
        static_num_cols=STATIC_NUM_COLS_DEFAULT,
        dynamic_cols=DYNAMIC_COLS_DEFAULT,
        cat_col=CAT_COL_DEFAULT,
    )
    if temporal_df.empty:
        raise ValueError("No temporal samples were created. Try a smaller --window-size.")
    print_temporal_dataset_summary(temporal_df, args.window_size)

    split_frames = {split_name: temporal_df[temporal_df["split"] == split_name].copy() for split_name in SPLIT_ORDER}

    rows = []
    for rep in range(args.repeats):
        seed_everything(RANDOM_SEED + rep)

        train_df = split_frames["train_us"].copy()
        overlap_df = split_frames["val_us_overlap"].copy()
        nonoverlap_df = split_frames["val_us_nonoverlap"].copy()
        external_df = split_frames["test_external"].copy()

        train_df, overlap_df, nonoverlap_df, n_cats, encoder = prepare_temporal_category(
            train_df, overlap_df, nonoverlap_df, CAT_COL_DEFAULT
        )
        train_df, overlap_df, combined_other, stats = prepare_temporal_numeric_features(
            train_df, overlap_df, external_df.copy(), STATIC_NUM_COLS_DEFAULT, DYNAMIC_COLS_DEFAULT
        )

        if len(nonoverlap_df) > 0:
            nonoverlap_df = nonoverlap_df.copy()
            nonoverlap_df[CAT_COL_DEFAULT] = encoder.transform(nonoverlap_df[CAT_COL_DEFAULT])
            nonoverlap_df = transform_temporal_numeric_features(
                nonoverlap_df,
                STATIC_NUM_COLS_DEFAULT,
                DYNAMIC_COLS_DEFAULT,
                stats,
            )
        external_df = combined_other.copy()
        if len(external_df) > 0:
            external_df[CAT_COL_DEFAULT] = encoder.transform(external_df[CAT_COL_DEFAULT])

        train_loader, overlap_loader, _ = make_temporal_dataloaders(
            train_df, overlap_df, external_df, STATIC_NUM_COLS_DEFAULT, CAT_COL_DEFAULT
        )
        nonoverlap_loader = make_temporal_dataloaders(train_df, overlap_df, nonoverlap_df, STATIC_NUM_COLS_DEFAULT, CAT_COL_DEFAULT)[2]
        external_loader = make_temporal_dataloaders(train_df, overlap_df, external_df, STATIC_NUM_COLS_DEFAULT, CAT_COL_DEFAULT)[2]

        split_eval = {
            "train_us": (train_df, train_loader),
            "val_us_overlap": (overlap_df, overlap_loader),
            "val_us_nonoverlap": (nonoverlap_df, nonoverlap_loader),
            "test_external": (external_df, external_loader),
        }

        model_specs = {
            "TemporalCNN_MLPStatic": (
                DualBranchTemporalCNNRegressor,
                {
                    "static_dim": len(STATIC_NUM_COLS_DEFAULT),
                    "dynamic_dim": len(DYNAMIC_COLS_DEFAULT),
                    "n_cats": n_cats,
                    "emb_dim": 8,
                    "static_hidden": 64,
                    "temporal_channels": 64,
                    "fusion_hidden": 128,
                    "dropout": 0.2,
                },
            ),
            "TemporalCNN_ResNetStatic": (
                DualBranchTemporalCNNResNetRegressor,
                {
                    "static_dim": len(STATIC_NUM_COLS_DEFAULT),
                    "dynamic_dim": len(DYNAMIC_COLS_DEFAULT),
                    "n_cats": n_cats,
                    "emb_dim": 8,
                    "static_width": 64,
                    "static_blocks": 2,
                    "temporal_channels": 64,
                    "fusion_hidden": 128,
                    "dropout": 0.2,
                },
            ),
            "TemporalCNN_TabNetStatic": (
                DualBranchTemporalCNNTabNetRegressor,
                {
                    "static_dim": len(STATIC_NUM_COLS_DEFAULT),
                    "dynamic_dim": len(DYNAMIC_COLS_DEFAULT),
                    "n_cats": n_cats,
                    "emb_dim": 8,
                    "static_hidden": 64,
                    "temporal_channels": 64,
                    "fusion_hidden": 128,
                    "dropout": 0.2,
                },
            ),
            "TemporalCNN_MoE": (
                DualBranchTemporalCNNMoERegressor,
                {
                    "static_dim": len(STATIC_NUM_COLS_DEFAULT),
                    "dynamic_dim": len(DYNAMIC_COLS_DEFAULT),
                    "n_cats": n_cats,
                    "emb_dim": 8,
                    "static_hidden": 64,
                    "temporal_channels": 64,
                    "n_experts": 4,
                    "expert_hidden": 128,
                    "gate_hidden": 64,
                    "dropout": 0.2,
                },
            ),
            "TemporalCNN_ResNetStatic_MoE": (
                DualBranchTemporalCNNResNetMoERegressor,
                {
                    "static_dim": len(STATIC_NUM_COLS_DEFAULT),
                    "dynamic_dim": len(DYNAMIC_COLS_DEFAULT),
                    "n_cats": n_cats,
                    "emb_dim": 8,
                    "static_width": 64,
                    "static_blocks": 2,
                    "temporal_channels": 64,
                    "n_experts": 4,
                    "expert_hidden": 128,
                    "gate_hidden": 64,
                    "dropout": 0.2,
                },
            ),
            "TemporalCNN_TabNetStatic_MoE": (
                DualBranchTemporalCNNTabNetMoERegressor,
                {
                    "static_dim": len(STATIC_NUM_COLS_DEFAULT),
                    "dynamic_dim": len(DYNAMIC_COLS_DEFAULT),
                    "n_cats": n_cats,
                    "emb_dim": 8,
                    "static_hidden": 64,
                    "temporal_channels": 64,
                    "n_experts": 4,
                    "expert_hidden": 128,
                    "gate_hidden": 64,
                    "dropout": 0.2,
                },
            ),
        }
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in selected_models if m not in model_specs]
        if unknown:
            raise ValueError(f"Unknown models in --models: {unknown}. Available: {sorted(model_specs)}")

        for model_name in selected_models:
            model_cls, model_kwargs = model_specs[model_name]
            print(f"[repeat {rep + 1}/{args.repeats}] training {model_name}")
            model = model_cls(**model_kwargs)
            trained = fit_model(model, train_loader, overlap_loader, device)

            artifact_dir = out_dir / f"{model_name}_rep{rep + 1}"
            save_temporal_cnn_artifact(
                artifact_dir=artifact_dir,
                model=trained,
                model_name=model_name,
                model_kwargs=model_kwargs,
                window_size=args.window_size,
                static_num_cols=STATIC_NUM_COLS_DEFAULT,
                dynamic_cols=DYNAMIC_COLS_DEFAULT,
                cat_col=CAT_COL_DEFAULT,
                stats=stats,
                encoder=encoder,
            )

            metrics_by_split = {}
            for split_name in SPLIT_ORDER:
                split_df, split_loader = split_eval[split_name]
                _, metrics = run_epoch(trained, split_loader, device)
                metrics_by_split[split_name] = metrics
                rows.append(
                    {
                        "scheme": "US_TRANSFER_TEMPORAL",
                        "scheme_desc": "US train/val + external world test with temporal models",
                        "model": model_name,
                        "repeat": rep + 1,
                        "split": split_name,
                        "n": len(split_df),
                        "window_size": args.window_size,
                        **metrics,
                    }
                )
            print_us_transfer_metrics(model_name, rep + 1, metrics_by_split)

    raw_path, summary_path = save_result_tables(rows, out_dir, "lfmc_us_transfer_temporal")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
