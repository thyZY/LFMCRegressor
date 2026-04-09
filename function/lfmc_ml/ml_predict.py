from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from common_features import Y_COL, eval_regression, load_json, save_json


class LabelEncoderWithUNK:
    def __init__(self):
        self.mapping: Dict[str, int] = {}
        self.unk_id: int = 0

    def fit(self, values: pd.Series) -> "LabelEncoderWithUNK":
        uniq = np.unique(values.astype(str).to_numpy())
        self.mapping = {v: i for i, v in enumerate(uniq)}
        self.unk_id = len(self.mapping)
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        return values.astype(str).map(lambda x: self.mapping.get(x, self.unk_id)).to_numpy(dtype=np.int32)

    def to_dict(self) -> Dict:
        return {"mapping": self.mapping, "unk_id": self.unk_id}

    @classmethod
    def from_dict(cls, d: Dict) -> "LabelEncoderWithUNK":
        obj = cls()
        obj.mapping = {str(k): int(v) for k, v in d["mapping"].items()}
        obj.unk_id = int(d["unk_id"])
        return obj


def _encode_category(train_df, val_df, test_df, cat_col) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoderWithUNK]:
    enc = LabelEncoderWithUNK().fit(train_df[cat_col])

    def _apply(df):
        x = df.copy()
        x[cat_col] = enc.transform(x[cat_col])
        return x

    return _apply(train_df), _apply(val_df), _apply(test_df), enc


def train_catboost(train_df, val_df, test_df, num_cols, cat_col, seed):
    feature_cols = num_cols + [cat_col]

    train_pool = Pool(train_df[feature_cols], train_df[Y_COL], cat_features=[cat_col], weight=train_df["sample_weight"])
    val_pool = Pool(val_df[feature_cols], val_df[Y_COL], cat_features=[cat_col], weight=val_df["sample_weight"])
    test_pool = Pool(test_df[feature_cols], test_df[Y_COL], cat_features=[cat_col], weight=test_df["sample_weight"])

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

    preds = {
        "train": model.predict(train_pool),
        "val": model.predict(val_pool),
        "test": model.predict(test_pool),
    }
    return model, preds, None


def train_lgbm(train_df, val_df, test_df, num_cols, cat_col, seed):
    train_df, val_df, test_df, enc = _encode_category(train_df, val_df, test_df, cat_col)
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

    preds = {
        "train": model.predict(x_train),
        "val": model.predict(x_val),
        "test": model.predict(x_test),
    }
    return model, preds, enc


def train_xgb(train_df, val_df, test_df, num_cols, cat_col, seed):
    train_df, val_df, test_df, enc = _encode_category(train_df, val_df, test_df, cat_col)
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

    preds = {
        "train": model.predict(x_train),
        "val": model.predict(x_val),
        "test": model.predict(x_test),
    }
    return model, preds, enc


def save_ml_artifact(artifact_dir: Path, model_name: str, model, num_cols, cat_col, encoder=None):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "CatBoost":
        model.save_model(str(artifact_dir / "model.cbm"))
    else:
        joblib.dump(model, artifact_dir / "model.joblib")

    meta = {
        "model_name": model_name,
        "num_cols": list(num_cols),
        "cat_col": cat_col,
        "encoder": encoder.to_dict() if encoder is not None else None,
    }
    save_json(meta, artifact_dir / "preprocess.json")


def load_ml_artifact(artifact_dir: Path):
    cfg = load_json(artifact_dir / "preprocess.json")
    model_name = cfg["model_name"]
    if model_name == "CatBoost":
        model = CatBoostRegressor()
        model.load_model(str(artifact_dir / "model.cbm"))
        encoder = None
    else:
        model = joblib.load(artifact_dir / "model.joblib")
        encoder = LabelEncoderWithUNK.from_dict(cfg["encoder"])
    return model_name, model, cfg["num_cols"], cfg["cat_col"], encoder


def predict_ml(model_name: str, model, df: pd.DataFrame, num_cols, cat_col, encoder=None) -> np.ndarray:
    feature_cols = list(num_cols) + [cat_col]
    x = df.copy()
    if model_name == "CatBoost":
        pool = Pool(x[feature_cols], cat_features=[cat_col])
        return model.predict(pool)

    x[cat_col] = encoder.transform(x[cat_col])
    return model.predict(x[feature_cols])