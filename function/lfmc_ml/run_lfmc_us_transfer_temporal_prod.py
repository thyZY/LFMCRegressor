from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common_features import DATE_COL, LAT_COL, build_prediction_dataframe_for_date
from lfmc_final_common import IGBP_COL, IGBP_TO_LC_DOM, LC_VEG_COLS, VOD_COLS
from lfmc_final_temporal_cnn_common import (
    DualBranchTemporalCNNMoERegressor,
    DualBranchTemporalCNNRegressor,
    DualBranchTemporalCNNResNetMoERegressor,
    DualBranchTemporalCNNResNetRegressor,
    DualBranchTemporalCNNTabNetMoERegressor,
    DualBranchTemporalCNNTabNetRegressor,
    LabelEncoderWithUNK,
)


MODEL_REGISTRY = {
    "DualBranchTemporalCNN": DualBranchTemporalCNNRegressor,
    "TemporalCNN_MLPStatic": DualBranchTemporalCNNRegressor,
    "TemporalCNN_MoE": DualBranchTemporalCNNMoERegressor,
    "TemporalCNN_ResNetStatic": DualBranchTemporalCNNResNetRegressor,
    "TemporalCNN_ResNetStatic_MoE": DualBranchTemporalCNNResNetMoERegressor,
    "TemporalCNN_TabNetStatic": DualBranchTemporalCNNTabNetRegressor,
    "TemporalCNN_TabNetStatic_MoE": DualBranchTemporalCNNTabNetMoERegressor,
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def build_base_inference_mask(base_df: pd.DataFrame, nonveg_threshold: float = 0.7) -> np.ndarray:
    if "valid_for_inference" in base_df.columns:
        infer_mask = base_df["valid_for_inference"].fillna(0).astype(np.uint8) == 1
    elif "predictor_valid" in base_df.columns:
        infer_mask = base_df["predictor_valid"].fillna(0).astype(np.uint8) == 1
    else:
        infer_mask = np.ones(len(base_df), dtype=bool)

    if "nonveg_frac" in base_df.columns:
        nonveg_frac = pd.to_numeric(base_df["nonveg_frac"], errors="coerce")
        infer_mask = infer_mask & ~(nonveg_frac > float(nonveg_threshold)).fillna(False)

    # For production over rasters, only pixels with good-quality VOD on the
    # target day are allowed to enter inference.
    if "VOD_QC" in base_df.columns:
        vod_qc = pd.to_numeric(base_df["VOD_QC"], errors="coerce")
        infer_mask = infer_mask & (vod_qc == 0).fillna(False)

    return np.asarray(infer_mask, dtype=bool)


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
        pd.DataFrame(
            {
                "row": base_df["row"].values,
                "col": base_df["col"].values,
                "LFMC_pred": y_pred.astype(np.float32),
            }
        ).to_csv(output_fp, index=False, encoding="utf-8-sig")
        return

    raise ValueError(f"Unsupported output format: {output_fp.suffix}")


def load_temporal_artifact(artifact_dir: Path, device: torch.device):
    ckpt = torch.load(artifact_dir / "model.pt", map_location=device)
    cfg = load_json(artifact_dir / "preprocess.json")

    model_name = ckpt.get("model_name", "DualBranchTemporalCNN")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported temporal model in artifact: {model_name}")

    model = MODEL_REGISTRY[model_name](**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    encoder = LabelEncoderWithUNK.from_dict(cfg["encoder"])
    return model_name, model, cfg, encoder


def build_temporal_prediction_inputs(
    date: str,
    window_size: int,
    static_num_cols: list[str],
    dynamic_cols: list[str],
    cat_col: str,
    nonveg_threshold: float,
):
    target_dt = pd.to_datetime(date)
    date_list = [target_dt - timedelta(days=i) for i in range(window_size - 1, -1, -1)]

    base_frames = []
    meta = None
    for dt in date_list:
        df_i, meta_i = build_prediction_dataframe_for_date(dt.strftime("%Y%m%d"))
        df_i = align_to_s6_lite_features(df_i)
        base_frames.append(df_i)
        if meta is None:
            meta = meta_i

    current_df = base_frames[-1].copy()
    infer_mask = build_base_inference_mask(current_df, nonveg_threshold=nonveg_threshold)

    for col in static_num_cols:
        infer_mask = infer_mask & current_df[col].notna().to_numpy(dtype=bool)
    infer_mask = infer_mask & current_df[cat_col].notna().to_numpy(dtype=bool)

    dyn_seq = np.empty((len(current_df), window_size, len(dynamic_cols)), dtype=np.float32)
    dyn_seq[:] = np.nan

    for t_idx, df_i in enumerate(base_frames):
        arr = df_i[dynamic_cols].to_numpy(dtype=np.float32)
        dyn_seq[:, t_idx, :] = arr

        non_vod_cols = [c for c in dynamic_cols if c not in VOD_COLS]
        if non_vod_cols:
            infer_mask = infer_mask & df_i[non_vod_cols].notna().all(axis=1).to_numpy(dtype=bool)

    static_num = current_df[static_num_cols].to_numpy(dtype=np.float32)
    cat_vals = current_df[cat_col].astype(str).to_numpy(dtype=object)
    return current_df, meta, infer_mask, static_num, cat_vals, dyn_seq


def transform_temporal_inputs(
    static_num: np.ndarray,
    dyn_seq: np.ndarray,
    cat_vals: np.ndarray,
    stats: dict,
    encoder: LabelEncoderWithUNK,
    dynamic_cols: list[str],
):
    static_mean = pd.Series(stats["static_mean"])
    static_std = pd.Series(stats["static_std"]).replace(0, 1.0)
    dyn_mean = np.asarray(stats["dynamic_mean"], dtype=np.float32)
    dyn_std = np.asarray(stats["dynamic_std"], dtype=np.float32)
    dyn_std[dyn_std == 0] = 1.0

    static_df = pd.DataFrame(static_num, columns=list(static_mean.index))
    static_arr = ((static_df - static_mean[static_df.columns]) / static_std[static_df.columns]).to_numpy(dtype=np.float32)

    dyn_arr = (dyn_seq - dyn_mean.reshape(1, 1, -1)) / dyn_std.reshape(1, 1, -1)
    vod_idx = [dynamic_cols.index(c) for c in dynamic_cols if c in VOD_COLS]
    if vod_idx:
        dyn_arr[:, :, vod_idx] = np.where(np.isnan(dyn_arr[:, :, vod_idx]), 0.0, dyn_arr[:, :, vod_idx])

    if not np.isfinite(static_arr).all():
        raise ValueError("Static inputs contain non-finite values after preprocessing.")
    if not np.isfinite(dyn_arr).all():
        raise ValueError("Dynamic inputs contain non-finite values after preprocessing.")

    cat_arr = encoder.transform(pd.Series(cat_vals.astype(str)))
    return static_arr, dyn_arr.astype(np.float32), cat_arr.astype(np.int64)


def predict_in_batches(
    model: torch.nn.Module,
    x_static: np.ndarray,
    x_cat: np.ndarray,
    x_dyn: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_static), batch_size):
            end = min(start + batch_size, len(x_static))
            xb_static = torch.tensor(x_static[start:end], dtype=torch.float32, device=device)
            xb_cat = torch.tensor(x_cat[start:end, None], dtype=torch.long, device=device)
            xb_dyn = torch.tensor(x_dyn[start:end], dtype=torch.float32, device=device)
            pred = model(xb_static, xb_cat, xb_dyn).squeeze(-1).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def predict_raster_array(
    artifact_dir: Path,
    date: str,
    nonveg_threshold: float = 0.7,
    batch_size: int = 131072,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, model, cfg, encoder = load_temporal_artifact(artifact_dir, device)

    current_df, meta, infer_mask, static_num, cat_vals, dyn_seq = build_temporal_prediction_inputs(
        date=date,
        window_size=int(cfg["window_size"]),
        static_num_cols=list(cfg["static_num_cols"]),
        dynamic_cols=list(cfg["dynamic_cols"]),
        cat_col=str(cfg["cat_col"]),
        nonveg_threshold=nonveg_threshold,
    )

    full_pred = np.full(len(current_df), np.nan, dtype=np.float32)
    if np.any(infer_mask):
        x_static, x_dyn, x_cat = transform_temporal_inputs(
            static_num[infer_mask],
            dyn_seq[infer_mask],
            cat_vals[infer_mask],
            cfg["stats"],
            encoder,
            list(cfg["dynamic_cols"]),
        )
        y_pred = predict_in_batches(model, x_static, x_cat, x_dyn, device, batch_size=batch_size)
        full_pred[infer_mask] = y_pred

    return model_name, full_pred, current_df, meta


def run_predict_raster(
    artifact_dir: Path,
    date: str,
    output_path: str,
    nonveg_threshold: float = 0.7,
    batch_size: int = 131072,
):
    model_name, full_pred, current_df, meta = predict_raster_array(
        artifact_dir=artifact_dir,
        date=date,
        nonveg_threshold=nonveg_threshold,
        batch_size=batch_size,
    )
    save_prediction_grid(full_pred, current_df, meta, output_path)
    print(f"[OK] {model_name} raster prediction finished -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--nonveg-threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=131072)
    args = parser.parse_args()

    run_predict_raster(
        artifact_dir=Path(args.artifact_dir),
        date=args.date,
        output_path=args.output_path,
        nonveg_threshold=args.nonveg_threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
