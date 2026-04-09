from __future__ import annotations

import argparse
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lfmc_batch_common import (
    OUT_DIR,
    RANDOM_SEED,
    SCHEMES,
    VOD_COLS,
    Y_COL,
    build_scheme_frame,
    eval_regression,
    load_base_dataframe,
    save_result_tables,
)


class LFMCSet(Dataset):
    def __init__(self, df, num_cols, cat_col):
        self.x_num = torch.tensor(df[num_cols].to_numpy(np.float32), dtype=torch.float32)
        self.x_cat = torch.tensor(df[[cat_col]].to_numpy(np.int64), dtype=torch.long)
        self.y = torch.tensor(df[Y_COL].to_numpy(np.float32), dtype=torch.float32)
        self.w = torch.tensor(df["sample_weight"].to_numpy(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx], self.w[idx]


class MLPRegressor(nn.Module):
    def __init__(self, num_dim, n_cats, emb_dim=8, hidden=(256, 128), dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)

        layers = []
        in_dim = num_dim + emb_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat[:, 0])], dim=1)
        return self.net(x)


class TabTransformerLite(nn.Module):
    def __init__(self, num_dim, n_cats, emb_dim=16, n_heads=4, n_layers=2, hidden=(256, 128), dropout=0.2):
        super().__init__()
        self.cat_emb = nn.Embedding(n_cats, emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        layers = []
        in_dim = num_dim + emb_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        cat_ctx = self.encoder(self.cat_emb(x_cat)).mean(dim=1)
        x = torch.cat([x_num, cat_ctx], dim=1)
        return self.head(x)


def weighted_mae(pred, y, w):
    err = torch.abs(pred.squeeze(-1) - y)
    return (err * w).sum() / w.sum().clamp_min(1e-8)


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    n = 0
    preds_all = []
    y_all = []

    for x_num, x_cat, y, w in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        w = w.to(device)

        pred = model(x_num, x_cat)
        loss = weighted_mae(pred, y, w)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(y)
        n += len(y)
        preds_all.append(pred.detach().cpu().numpy().reshape(-1))
        y_all.append(y.detach().cpu().numpy().reshape(-1))

    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    return total_loss / max(n, 1), eval_regression(y_all, preds_all)


def fit_model(model, train_loader, val_loader, device, max_epochs=80, patience=10):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_state = None
    best_val = float("inf")
    wait = 0

    for _ in range(max_epochs):
        run_epoch(model, train_loader, device, optimizer)
        val_loss, _ = run_epoch(model, val_loader, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _prepare_dl_numeric_features(train_df, val_df, test_df, num_cols):
    """
    Deep learning specific numeric preprocessing:
    1. Keep the strict non-VOD filtering already done in lfmc_batch_common.
    2. Allow partial VOD missingness.
    3. Standardize using train statistics.
    4. Fill only VOD NaNs with 0 after standardization, meaning train-mean in standardized space.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    vod_cols_in_use = [c for c in VOD_COLS if c in num_cols]

    mean = train_df[num_cols].mean()
    std = train_df[num_cols].std().replace(0, 1.0)

    for df_ in [train_df, val_df, test_df]:
        df_[num_cols] = (df_[num_cols] - mean) / std

        # Only VOD columns are allowed to remain missing at this stage.
        if vod_cols_in_use:
            df_[vod_cols_in_use] = df_[vod_cols_in_use].fillna(0.0)

    # Final safety check: after VOD filling, no numeric NaN/inf should remain.
    for split_name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        arr = df_[num_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(arr).all():
            bad_mask = ~np.isfinite(arr)
            bad_count = int(bad_mask.sum())
            raise ValueError(f"{split_name} numeric features still contain {bad_count} non-finite values after preprocessing.")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_df, igbp_num_cols = load_base_dataframe()
    rows = []

    for scheme in SCHEMES:
        print(f"\n=== Running {scheme.code} | {scheme.description} ===")

        scheme_df, num_cols, cat_col = build_scheme_frame(base_df, scheme, igbp_num_cols)

        train_df = scheme_df[scheme_df["split"] == "train"].copy()
        val_df = scheme_df[scheme_df["split"] == "val"].copy()
        test_df = scheme_df[scheme_df["split"] == "test"].copy()

        # Encode categorical feature
        all_cats = pd.concat([train_df[cat_col], val_df[cat_col], test_df[cat_col]], axis=0).astype(str).unique()
        cat_map = {v: i for i, v in enumerate(all_cats)}

        for df_ in [train_df, val_df, test_df]:
            df_[cat_col] = df_[cat_col].astype(str).map(cat_map).astype(np.int64)

        # DL-specific numeric preprocessing
        train_df, val_df, test_df = _prepare_dl_numeric_features(train_df, val_df, test_df, num_cols)

        train_loader = DataLoader(LFMCSet(train_df, num_cols, cat_col), batch_size=512, shuffle=True)
        val_loader = DataLoader(LFMCSet(val_df, num_cols, cat_col), batch_size=1024, shuffle=False)
        test_loader = DataLoader(LFMCSet(test_df, num_cols, cat_col), batch_size=1024, shuffle=False)

        num_dim = len(num_cols)
        n_cats = len(cat_map)

        for rep in range(args.repeats):
            seed = RANDOM_SEED + rep
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            models = {
                "MLP": MLPRegressor(num_dim=num_dim, n_cats=n_cats),
                "TabTransformer": TabTransformerLite(num_dim=num_dim, n_cats=n_cats),
            }

            for model_name, model in models.items():
                print(f"  [{scheme.code}] {model_name} repeat {rep + 1}/{args.repeats}")
                trained = fit_model(model, train_loader, val_loader, device)

                train_loss, train_metrics = run_epoch(trained, train_loader, device)
                val_loss, val_metrics = run_epoch(trained, val_loader, device)
                test_loss, test_metrics = run_epoch(trained, test_loader, device)

                for split_name, split_df, metrics in [
                    ("train", train_df, train_metrics),
                    ("val", val_df, val_metrics),
                    ("test", test_df, test_metrics),
                ]:
                    rows.append(
                        {
                            "scheme": scheme.code,
                            "scheme_desc": scheme.description,
                            "model": model_name,
                            "repeat": rep + 1,
                            "split": split_name,
                            "n": len(split_df),
                            **metrics,
                        }
                    )

    raw_path, summary_path = save_result_tables(rows, OUT_DIR, "lfmc_7schemes_dl")
    print(f"Saved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    main()
