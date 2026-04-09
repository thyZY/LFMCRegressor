from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dl_models import (
    MODEL_DEFAULT_KWARGS,
    MODEL_REGISTRY,
    fit_model,
    make_dataloaders,
    prepare_dl_category,
    prepare_dl_numeric_features,
    run_epoch,
    save_dl_artifact,
    seed_everything,
    transform_numeric_features,
)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=r"d:\Python\jupyter\jupyter\LFMCRegressor\notebooks\artifacts\dl_us_transfer")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--models", type=str, default="ResNetMLP,TabNet,MoE_ResNetMLP")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_df, num_cols, cat_col = build_us_transfer_s6_lite_frame(args.data_path)
    print("Device:", device)
    print_us_transfer_dataset_summary(final_df, num_cols, cat_col)
    split_frames = split_frame_by_eval_group(final_df)
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected_models if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models in --models: {unknown}. Available: {sorted(MODEL_REGISTRY)}")

    rows = []
    for rep in range(args.repeats):
        seed_everything(RANDOM_SEED + rep)

        train_df = split_frames["train_us"].copy()
        overlap_df = split_frames["val_us_overlap"].copy()
        nonoverlap_df = split_frames["val_us_nonoverlap"].copy()
        external_df = split_frames["test_external"].copy()

        train_df, overlap_df, nonoverlap_df, n_cats, encoder = prepare_dl_category(
            train_df, overlap_df, nonoverlap_df, cat_col
        )
        combined_test = external_df.copy()
        train_df, overlap_df, combined_test, stats = prepare_dl_numeric_features(
            train_df, overlap_df, combined_test, num_cols
        )

        if len(nonoverlap_df) > 0:
            nonoverlap_df = nonoverlap_df.copy()
            nonoverlap_df[cat_col] = encoder.transform(nonoverlap_df[cat_col])
            nonoverlap_df = transform_numeric_features(nonoverlap_df, num_cols, stats)

        if len(external_df) > 0:
            external_df = external_df.copy()
            external_df[cat_col] = encoder.transform(external_df[cat_col])
            external_df = transform_numeric_features(external_df, num_cols, stats)

        train_loader, overlap_loader, _ = make_dataloaders(
            train_df, overlap_df, external_df, num_cols, cat_col
        )
        nonoverlap_loader = make_dataloaders(train_df, overlap_df, nonoverlap_df, num_cols, cat_col)[2]
        external_loader = make_dataloaders(train_df, overlap_df, external_df, num_cols, cat_col)[2]

        split_eval = {
            "train_us": (train_df, train_loader),
            "val_us_overlap": (overlap_df, overlap_loader),
            "val_us_nonoverlap": (nonoverlap_df, nonoverlap_loader),
            "test_external": (external_df, external_loader),
        }

        for model_name in selected_models:
            model_cls = MODEL_REGISTRY[model_name]
            print(f"[repeat {rep + 1}/{args.repeats}] training {model_name}")
            model_kwargs = {"num_dim": len(num_cols), "n_cats": n_cats, **MODEL_DEFAULT_KWARGS[model_name]}
            model = model_cls(**model_kwargs)
            trained = fit_model(model, train_loader, overlap_loader, device)

            artifact_dir = out_dir / f"{model_name}_rep{rep + 1}"
            save_dl_artifact(
                artifact_dir,
                model_name,
                trained,
                model_kwargs,
                num_cols,
                cat_col,
                stats,
                encoder,
            )

            metrics_by_split = {}
            for split_name in SPLIT_ORDER:
                split_df, split_loader = split_eval[split_name]
                _, metrics = run_epoch(trained, split_loader, device)
                metrics_by_split[split_name] = metrics
                rows.append(
                    {
                        "scheme": "US_TRANSFER_S6_LITE",
                        "scheme_desc": "US train/val + external world test on S6-lite",
                        "model": model_name,
                        "repeat": rep + 1,
                        "split": split_name,
                        "n": len(split_df),
                        **metrics,
                    }
                )
            print_us_transfer_metrics(model_name, rep + 1, metrics_by_split)

    raw_path, summary_path = save_result_tables(rows, out_dir, "lfmc_us_transfer_dl")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
