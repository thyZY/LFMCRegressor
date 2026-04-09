from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_lfmc_us_transfer_temporal_prod import predict_raster_array, save_prediction_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir-a", type=str, required=True)
    parser.add_argument("--artifact-dir-b", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--nonveg-threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=131072)
    args = parser.parse_args()

    model_a, pred_a, base_df_a, meta_a = predict_raster_array(
        artifact_dir=Path(args.artifact_dir_a),
        date=args.date,
        nonveg_threshold=args.nonveg_threshold,
        batch_size=args.batch_size,
    )
    model_b, pred_b, base_df_b, meta_b = predict_raster_array(
        artifact_dir=Path(args.artifact_dir_b),
        date=args.date,
        nonveg_threshold=args.nonveg_threshold,
        batch_size=args.batch_size,
    )

    if len(base_df_a) != len(base_df_b):
        raise ValueError("The two ensemble predictions are not aligned.")

    pred_stack = np.vstack([pred_a, pred_b]).astype(np.float32)
    pred_mean = np.nanmean(pred_stack, axis=0).astype(np.float32)
    all_nan = np.isnan(pred_stack).all(axis=0)
    pred_mean[all_nan] = np.nan

    save_prediction_grid(pred_mean, base_df_a, meta_a, args.output_path)
    print(f"[OK] Ensemble raster prediction finished -> {args.output_path}")
    print(f"  model A: {model_a}")
    print(f"  model B: {model_b}")


if __name__ == "__main__":
    main()
