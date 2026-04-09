# -*- coding: utf-8 -*-
"""
Product-style ML inference entrypoint aligned to S6-lite features.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    CatBoostRegressor = None
    Pool = None

from common_features import build_prediction_dataframe_for_date, load_base_dataframe
from lfmc_final_common import (
    CORE_NUM_COLS_LITE,
    DATE_COL,
    IGBP_COL,
    IGBP_TO_LC_DOM,
    LAT_COL,
    LC_VEG_COLS,
    SEASON_COLS,
    VOD_COLS,
)


def _load_json_if_exists(fp: Path):
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_pickle_like(fp: Path):
    if not fp.exists():
        return None

    if joblib is not None and fp.suffix.lower() in [".joblib", ".pkl", ".pickle"]:
        try:
            return joblib.load(fp)
        except Exception:
            pass

    with open(fp, "rb") as f:
        return pickle.load(f)


def load_model_artifact(model_name: str, artifact_dir: Path):
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact_dir does not exist: {artifact_dir}")

    preprocess = _load_json_if_exists(artifact_dir / "preprocess.json")
    if preprocess is not None:
        model_name_loaded = preprocess["model_name"]
        if model_name_loaded != model_name:
            raise ValueError(
                f"artifact model_name={model_name_loaded} != requested model_name={model_name}"
            )

        model = None
        model_path = None

        cbm_fp = artifact_dir / "model.cbm"
        if cbm_fp.exists():
            if CatBoostRegressor is None:
                raise ImportError("CatBoost is required to load model.cbm artifacts.")
            model = CatBoostRegressor()
            model.load_model(str(cbm_fp))
            model_path = cbm_fp

        if model is None:
            for fn in ["model.joblib", "model.pkl", "model.pickle"]:
                fp = artifact_dir / fn
                if fp.exists():
                    model = _load_pickle_like(fp)
                    model_path = fp
                    break

        if model is None:
            raise FileNotFoundError(f"No usable model file found in {artifact_dir}")

        return {
            "model": model,
            "model_path": model_path,
            "num_cols": list(preprocess["num_cols"]),
            "cat_col": preprocess["cat_col"],
            "encoder": preprocess.get("encoder"),
            "artifact_format": "preprocess_json",
        }

    model = None
    model_path = None

    cbm_fp = artifact_dir / "model.cbm"
    if cbm_fp.exists():
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is required to load model.cbm artifacts.")
        model = CatBoostRegressor()
        model.load_model(str(cbm_fp))
        model_path = cbm_fp

    if model is None:
        for fn in ["model.joblib", "model.pkl", "model.pickle"]:
            fp = artifact_dir / fn
            if fp.exists():
                model = _load_pickle_like(fp)
                model_path = fp
                break

    if model is None:
        raise FileNotFoundError(
            f"No usable model file found in {artifact_dir} (model.cbm / model.joblib / model.pkl)."
        )

    feature_columns = None
    meta = _load_json_if_exists(artifact_dir / "model_meta.json")
    if meta is not None and "feature_columns" in meta:
        feature_columns = meta["feature_columns"]

    feature_cols_json = _load_json_if_exists(artifact_dir / "feature_columns.json")
    if feature_cols_json is not None:
        if isinstance(feature_cols_json, list):
            feature_columns = feature_cols_json
        elif isinstance(feature_cols_json, dict) and "feature_columns" in feature_cols_json:
            feature_columns = feature_cols_json["feature_columns"]

    encoders = None
    for fn in ["encoders.joblib", "encoders.pkl", "encoders.pickle"]:
        fp = artifact_dir / fn
        if fp.exists():
            encoders = _load_pickle_like(fp)
            break

    return {
        "model": model,
        "model_path": model_path,
        "feature_columns": feature_columns,
        "encoders": encoders,
        "num_cols": None,
        "cat_col": None,
        "encoder": None,
        "artifact_format": "generic",
    }


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


def build_inference_mask(
    base_df: pd.DataFrame,
    num_cols: list[str],
    cat_col: str,
    nonveg_threshold: float = 0.7,
) -> np.ndarray:
    if "valid_for_inference" in base_df.columns:
        infer_mask = base_df["valid_for_inference"].fillna(0).astype(np.uint8) == 1
    elif "predictor_valid" in base_df.columns:
        infer_mask = base_df["predictor_valid"].fillna(0).astype(np.uint8) == 1
    else:
        infer_mask = np.ones(len(base_df), dtype=bool)

    if "nonveg_frac" in base_df.columns:
        nonveg_frac = pd.to_numeric(base_df["nonveg_frac"], errors="coerce")
        infer_mask = infer_mask & ~(nonveg_frac > float(nonveg_threshold)).fillna(False)

    strict_num_cols = [c for c in num_cols if c not in VOD_COLS]
    if strict_num_cols:
        infer_mask = infer_mask & base_df[strict_num_cols].notna().all(axis=1).to_numpy(dtype=bool)

    infer_mask = infer_mask & base_df[cat_col].notna().to_numpy(dtype=bool)
    return np.asarray(infer_mask, dtype=bool)


def fill_vod_missing_for_prediction(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    vod_cols_in_use = [c for c in VOD_COLS if c in feature_columns and c in out.columns]
    if vod_cols_in_use:
        out[vod_cols_in_use] = out[vod_cols_in_use].fillna(0.0)
    return out


def prepare_features_for_prediction(base_df: pd.DataFrame, artifact_bundle: dict):
    num_cols = artifact_bundle.get("num_cols")
    cat_col = artifact_bundle.get("cat_col")

    if num_cols is not None and cat_col is not None:
        feature_columns = list(num_cols) + [cat_col]
        missing_cols = [c for c in feature_columns if c not in base_df.columns]
        if missing_cols:
            raise ValueError(f"Prediction input is missing required features: {missing_cols}")

        X = fill_vod_missing_for_prediction(base_df[feature_columns].copy(), feature_columns)
        X[cat_col] = X[cat_col].astype(str).fillna("Unknown")
        return X, feature_columns

    feature_columns = artifact_bundle["feature_columns"]
    if feature_columns is None:
        raise ValueError("Generic artifact prediction requires feature_columns metadata.")

    missing_cols = [c for c in feature_columns if c not in base_df.columns]
    if missing_cols:
        raise ValueError(f"Prediction input is missing required features: {missing_cols}")

    X = fill_vod_missing_for_prediction(base_df[feature_columns].copy(), feature_columns)
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in obj_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    encoders = artifact_bundle.get("encoders")
    if encoders is not None and isinstance(encoders, dict):
        for col, enc in encoders.items():
            if col in X.columns:
                X[col] = enc.transform(X[col].astype(str).fillna("Unknown"))

    return X, feature_columns


def predict_with_artifact(model_name: str, artifact_bundle: dict, pred_df: pd.DataFrame) -> np.ndarray:
    X, feature_cols = prepare_features_for_prediction(pred_df, artifact_bundle)

    if artifact_bundle.get("artifact_format") == "preprocess_json":
        encoder_cfg = artifact_bundle.get("encoder")
        cat_col = artifact_bundle["cat_col"]

        if model_name != "CatBoost" and encoder_cfg is not None:
            mapping = {str(k): int(v) for k, v in encoder_cfg.get("mapping", {}).items()}
            unk_id = int(encoder_cfg.get("unk_id", len(mapping)))
            X[cat_col] = X[cat_col].map(lambda v: mapping.get(str(v), unk_id)).astype(np.int32)

        if model_name == "CatBoost":
            if Pool is None:
                raise ImportError("CatBoost Pool is unavailable.")
            pool = Pool(X[feature_cols], cat_features=[cat_col])
            y_pred = artifact_bundle["model"].predict(pool)
        else:
            y_pred = artifact_bundle["model"].predict(X[feature_cols])

        return np.asarray(y_pred, dtype=np.float32).reshape(-1)

    y_pred = artifact_bundle["model"].predict(X)
    return np.asarray(y_pred, dtype=np.float32).reshape(-1)


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
        try:
            import scipy.io as sio
            sio.savemat(str(output_fp), {"LFMC_pred": pred_arr})
            return
        except Exception as e:
            raise RuntimeError(f"保存 mat 失败: {e}")

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
            {"row": base_df["row"].values, "col": base_df["col"].values, "LFMC_pred": y_pred.astype(np.float32)}
        )
        tmp.to_csv(output_fp, index=False, encoding="utf-8-sig")
        return

    raise ValueError(f"Unsupported output format: {output_fp.suffix}")


def run_train(model_name: str, repeats: int = 1):
    script = Path(__file__).resolve().parent / "run_lfmc_final_ml_batch.py"
    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}")

    cmd = ["python", str(script), "--repeats", str(repeats)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_predict_csv(
    model_name: str,
    artifact_dir: Path,
    predict_path: str,
    output_path: str,
    nonveg_threshold: float = 0.7,
):
    base_df = load_base_dataframe(predict_path, require_target=False)
    base_df = align_to_s6_lite_features(base_df)
    artifact_bundle = load_model_artifact(model_name, artifact_dir)

    num_cols = artifact_bundle["num_cols"]
    cat_col = artifact_bundle["cat_col"]
    infer_mask = build_inference_mask(base_df, num_cols, cat_col, nonveg_threshold=nonveg_threshold)

    pred_df = base_df.loc[infer_mask].copy()
    y_pred = predict_with_artifact(model_name, artifact_bundle, pred_df)

    full_pred = np.full(len(base_df), np.nan, dtype=np.float32)
    full_pred[infer_mask] = y_pred

    save_predictions_table(base_df, full_pred, output_path)
    print(f"[OK] CSV prediction finished -> {output_path}")


def run_predict_raster(
    model_name: str,
    artifact_dir: Path,
    date: str,
    output_path: str,
    nonveg_threshold: float = 0.7,
):
    base_df, meta = build_prediction_dataframe_for_date(date)
    base_df = align_to_s6_lite_features(base_df)
    artifact_bundle = load_model_artifact(model_name, artifact_dir)

    num_cols = artifact_bundle["num_cols"]
    cat_col = artifact_bundle["cat_col"]
    infer_mask = build_inference_mask(base_df, num_cols, cat_col, nonveg_threshold=nonveg_threshold)

    pred_df = base_df.loc[infer_mask].copy()
    y_pred = predict_with_artifact(model_name, artifact_bundle, pred_df)

    save_prediction_grid(y_pred, pred_df, meta, output_path)
    print(f"[OK] Raster prediction finished -> {output_path}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--model-name", type=str, default="CatBoost")
    parser.add_argument("--artifact-dir", type=str, default=r".\artifacts\ml\CatBoost_rep1")
    parser.add_argument("--predict-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default="ml_predictions.csv")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nonveg-threshold", type=float, default=0.7)
    parser.add_argument("--input-mode", choices=["csv", "raster"], default="csv")
    parser.add_argument("--date", type=str, default=None, help="raster mode date, e.g. 20200115")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.model_name, repeats=args.repeats)
        return

    if args.input_mode == "csv":
        if args.predict_path is None:
            raise ValueError("csv mode requires --predict-path")
        run_predict_csv(
            args.model_name,
            Path(args.artifact_dir),
            args.predict_path,
            args.output_path,
            args.nonveg_threshold,
        )
        return

    if args.input_mode == "raster":
        if args.date is None:
            raise ValueError("raster mode requires --date, e.g. 20200115")
        run_predict_raster(
            args.model_name,
            Path(args.artifact_dir),
            args.date,
            args.output_path,
            args.nonveg_threshold,
        )
        return


if __name__ == "__main__":
    main()
