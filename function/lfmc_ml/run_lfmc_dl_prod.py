from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common_features import (
    DATE_COL,
    LAT_COL,
    build_prediction_dataframe_for_date,
    load_base_dataframe as load_prod_base_dataframe,
)
from dl_models import (
    MODEL_DEFAULT_KWARGS,
    MODEL_REGISTRY,
    fit_model,
    load_dl_artifact,
    make_dataloaders,
    prepare_dl_category,
    prepare_dl_numeric_features,
    run_epoch,
    save_dl_artifact,
    seed_everything,
    transform_numeric_features,
)
from lfmc_final_common import (
    IGBP_COL,
    IGBP_TO_LC_DOM,
    LC_VEG_COLS,
    RANDOM_SEED,
    VOD_COLS,
    build_final_s6_lite_frame,
    load_base_dataframe as load_train_base_dataframe,
    save_result_tables,
)


DEFAULT_OUT_DIR = Path(r"d:\Python\jupyter\jupyter\LFMCRegressor\notebooks\artifacts\dl")
DEFAULT_SCHEME = "FINAL_S6_LITE_PROD"
DEFAULT_SCHEME_DESC = "lc_dom + VOD + LAI/Hveg/LST + doy_sin/doy_cos/lat_norm + nonveg_skip"


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


def align_to_s6_lite_features(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()

    dt = pd.to_datetime(df[DATE_COL], errors="coerce")
    doy = dt.dt.dayofyear.astype(np.float32)
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)
    df["lat_norm"] = (pd.to_numeric(df[LAT_COL], errors="coerce") / 90.0).astype(np.float32)

    lc_mat = np.nan_to_num(
        df[LC_VEG_COLS].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    lc_mat = np.clip(lc_mat, 0.0, None)
    lc_idx = np.argmax(lc_mat, axis=1)
    lc_sum = lc_mat.sum(axis=1, keepdims=True)
    lc_sum[lc_sum == 0] = np.nan
    lc_norm = lc_mat / lc_sum

    lc_names = np.array(LC_VEG_COLS, dtype=object)
    df["lc_dom"] = pd.Categorical(lc_names[lc_idx], categories=LC_VEG_COLS)
    df["lc_dom_frac"] = lc_norm[np.arange(lc_norm.shape[0]), lc_idx].astype(np.float32)
    if IGBP_COL in df.columns:
        df["igbp_to_lc_dom"] = df[IGBP_COL].astype("string").map(IGBP_TO_LC_DOM).fillna("Unknown")

    return df


def build_inference_mask(base_df: pd.DataFrame, nonveg_threshold: float = 0.7) -> np.ndarray:
    if "valid_for_inference" in base_df.columns:
        infer_mask = base_df["valid_for_inference"].fillna(0).astype(np.uint8) == 1
    elif "predictor_valid" in base_df.columns:
        infer_mask = base_df["predictor_valid"].fillna(0).astype(np.uint8) == 1
    else:
        infer_mask = np.ones(len(base_df), dtype=bool)

    if "nonveg_frac" in base_df.columns:
        nonveg_frac = pd.to_numeric(base_df["nonveg_frac"], errors="coerce")
        infer_mask = infer_mask & ~(nonveg_frac > float(nonveg_threshold)).fillna(False)

    return np.asarray(infer_mask, dtype=bool)


def predict_dl_in_batches(
    model: torch.nn.Module,
    valid_df: pd.DataFrame,
    num_cols: list[str],
    cat_col: str,
    stats: dict,
    encoder,
    device: torch.device,
    batch_size: int = 262144,
) -> np.ndarray:
    if len(valid_df) == 0:
        return np.empty((0,), dtype=np.float32)

    x_df = valid_df.copy()
    x_df[cat_col] = encoder.transform(x_df[cat_col].astype(str))
    x_df = transform_numeric_features(x_df, num_cols, stats)

    x_num = x_df[num_cols].to_numpy(dtype=np.float32)
    x_cat = x_df[[cat_col]].to_numpy(dtype=np.int64)

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_df), batch_size):
            end = min(start + batch_size, len(x_df))
            xb_num = torch.tensor(x_num[start:end], dtype=torch.float32, device=device)
            xb_cat = torch.tensor(x_cat[start:end], dtype=torch.long, device=device)
            pred = model(xb_num, xb_cat).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)

    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def build_required_numeric_mask(base_df: pd.DataFrame, num_cols: list[str]) -> np.ndarray:
    # For S6-Lite inference, non-VOD numeric features must be present.
    # VOD columns are allowed to be partially missing and will be imputed later.
    vod_cols = [c for c in num_cols if c.startswith("VOD_")]
    required_cols = [c for c in num_cols if c not in vod_cols]
    if not required_cols:
        return np.ones(len(base_df), dtype=bool)
    return base_df[required_cols].notna().all(axis=1).to_numpy(dtype=bool)


def save_predictions_table(base_df: pd.DataFrame, y_pred: np.ndarray, output_path: str):
    output_fp = Path(output_path)
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    out_df = base_df.copy()
    out_df["LFMC_pred"] = y_pred.astype(np.float32)
    out_df.to_csv(output_fp, index=False, encoding="utf-8-sig")


def save_prediction_grid(y_pred: np.ndarray, base_df: pd.DataFrame, meta: dict, output_path: str):
    output_fp = Path(output_path)
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    pred_arr = np.full(meta["shape"], np.nan, dtype=np.float32)
    rr = base_df["row"].to_numpy(dtype=np.int32)
    cc = base_df["col"].to_numpy(dtype=np.int32)
    pred_arr[rr, cc] = y_pred.astype(np.float32)

    ext = output_fp.suffix.lower()

    if ext == ".npy":
        np.save(output_fp, pred_arr)
        return

    if ext == ".mat":
        import scipy.io as sio

        sio.savemat(str(output_fp), {"LFMC_pred": pred_arr})
        return

    if ext in [".h5", ".hdf5"]:
        import h5py

        with h5py.File(output_fp, "w") as f:
            f.create_dataset("LFMC_pred", data=pred_arr, compression="gzip")
            f.attrs["rows"] = meta["shape"][0]
            f.attrs["cols"] = meta["shape"][1]
            f.attrs["crs"] = meta.get("crs", "EPSG:4326")
            f.attrs["transform"] = np.array(
                meta.get("transform", (-180.0, 0.1, 0.0, 90.0, 0.0, -0.1)),
                dtype=np.float64,
            )
        return

    if ext in [".tif", ".tiff"]:
        try:
            import rasterio
            from rasterio.transform import from_origin
        except Exception as e:
            raise RuntimeError(f"保存 tif 失败: {e}")

        transform = from_origin(-180.0, 90.0, 0.1, 0.1)
        with rasterio.open(
            output_fp,
            "w",
            driver="GTiff",
            height=pred_arr.shape[0],
            width=pred_arr.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=np.nan,
            compress="lzw",
        ) as dst:
            dst.write(pred_arr, 1)
        return

    if ext == ".csv":
        tmp = pd.DataFrame(
            {
                "row": base_df["row"].values,
                "col": base_df["col"].values,
                "LFMC_pred": y_pred.astype(np.float32),
            }
        )
        tmp.to_csv(output_fp, index=False, encoding="utf-8-sig")
        return

    raise ValueError(f"Unsupported output format: {output_fp.suffix}")


def run_predict_csv(
    model_name: str,
    artifact_dir: Path,
    predict_path: str,
    output_path: str,
    nonveg_threshold: float = 0.7,
    batch_size: int = 262144,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_df = load_prod_base_dataframe(predict_path, require_target=False)
    base_df = align_to_s6_lite_features(base_df)

    model_name_loaded, model, num_cols, cat_col, stats, encoder = load_dl_artifact(artifact_dir, device)
    if model_name_loaded != model_name:
        raise ValueError(f"artifact model_name={model_name_loaded} != requested model_name={model_name}")

    infer_mask = build_inference_mask(base_df, nonveg_threshold=nonveg_threshold)
    infer_mask = infer_mask & build_required_numeric_mask(base_df, num_cols)
    valid_df = base_df.loc[infer_mask].copy()
    y_pred = predict_dl_in_batches(
        model, valid_df, num_cols, cat_col, stats, encoder, device, batch_size=batch_size
    )

    full_pred = np.full(len(base_df), np.nan, dtype=np.float32)
    full_pred[infer_mask] = y_pred

    save_predictions_table(base_df, full_pred, output_path)
    print(f"Saved predictions to: {output_path}")


def run_predict_raster(
    model_name: str,
    artifact_dir: Path,
    date: str,
    output_path: str,
    nonveg_threshold: float = 0.7,
    batch_size: int = 262144,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_df, meta = build_prediction_dataframe_for_date(date)
    base_df = align_to_s6_lite_features(base_df)

    model_name_loaded, model, num_cols, cat_col, stats, encoder = load_dl_artifact(artifact_dir, device)
    if model_name_loaded != model_name:
        raise ValueError(f"artifact model_name={model_name_loaded} != requested model_name={model_name}")

    infer_mask = build_inference_mask(base_df, nonveg_threshold=nonveg_threshold)
    infer_mask = infer_mask & build_required_numeric_mask(base_df, num_cols)
    valid_df = base_df.loc[infer_mask].copy()
    y_pred = predict_dl_in_batches(
        model, valid_df, num_cols, cat_col, stats, encoder, device, batch_size=batch_size
    )

    save_prediction_grid(y_pred, valid_df, meta, output_path)
    print(f"[OK] Raster prediction finished -> {output_path}")


def run_train(*args, **kwargs):
    repeats = int(kwargs["repeats"])
    out_dir = Path(kwargs["out_dir"])
    data_path = kwargs.get("data_path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_df = load_train_base_dataframe(Path(data_path) if data_path else None)
    final_df, num_cols, cat_col = build_final_s6_lite_frame(base_df)

    train_base = final_df[final_df["split"] == "train"].copy()
    val_base = final_df[final_df["split"] == "val"].copy()
    test_base = final_df[final_df["split"] == "test"].copy()
    num_dim = len(num_cols)

    print("Device:", device)
    print("=== Final S6-lite DL product dataset ===")
    print(f"rows: {len(final_df)}")
    print(f"train={len(train_base)}, val={len(val_base)}, test={len(test_base)}")
    print(f"num features: {len(num_cols) + 1} (numeric={len(num_cols)}, cat=1)")
    print("numeric columns:", num_cols)
    print("categorical column:", cat_col)

    rows = []

    for rep in range(repeats):
        seed = RANDOM_SEED + rep
        seed_everything(seed)

        train_df = train_base.copy()
        val_df = val_base.copy()
        test_df = test_base.copy()

        train_df, val_df, test_df, n_cats, encoder = prepare_dl_category(
            train_df, val_df, test_df, cat_col
        )
        train_df, val_df, test_df, stats = prepare_dl_numeric_features(
            train_df, val_df, test_df, num_cols
        )

        train_loader, val_loader, test_loader = make_dataloaders(
            train_df, val_df, test_df, num_cols, cat_col
        )

        for model_name, model_cls in MODEL_REGISTRY.items():
            print(f"[repeat {rep + 1}/{repeats}] training {model_name}")
            model_kwargs = {"num_dim": num_dim, "n_cats": n_cats, **MODEL_DEFAULT_KWARGS[model_name]}
            model = model_cls(**model_kwargs)
            trained = fit_model(model, train_loader, val_loader, device)

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

            _, train_metrics = run_epoch(trained, train_loader, device)
            _, val_metrics = run_epoch(trained, val_loader, device)
            _, test_metrics = run_epoch(trained, test_loader, device)
            print_split_metrics(
                model_name,
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
                        "scheme": DEFAULT_SCHEME,
                        "scheme_desc": DEFAULT_SCHEME_DESC,
                        "model": model_name,
                        "repeat": rep + 1,
                        "split": split_name,
                        "n": len(split_df),
                        **metrics,
                    }
                )

    raw_path, summary_path = save_result_tables(rows, out_dir, "lfmc_final_s6_lite_dl")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="TabNet")
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--predict-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="dl_predictions.csv")
    parser.add_argument("--input-mode", choices=["csv", "raster"], default="csv")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--nonveg-threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=262144)
    args = parser.parse_args()

    if args.mode == "train":
        run_train(repeats=args.repeats, out_dir=Path(args.out_dir), data_path=args.data_path)
        return

    if not args.artifact_dir:
        raise ValueError("predict mode requires --artifact-dir")

    if args.input_mode == "csv":
        if not args.predict_path:
            raise ValueError("csv mode requires --predict-path")
        run_predict_csv(
            args.model_name,
            Path(args.artifact_dir),
            args.predict_path,
            args.output_path,
            nonveg_threshold=args.nonveg_threshold,
            batch_size=args.batch_size,
        )
        return

    if args.input_mode == "raster":
        if not args.date:
            raise ValueError("raster mode requires --date, e.g. 20200115")
        run_predict_raster(
            args.model_name,
            Path(args.artifact_dir),
            args.date,
            args.output_path,
            nonveg_threshold=args.nonveg_threshold,
            batch_size=args.batch_size,
        )
        return


if __name__ == "__main__":
    main()
