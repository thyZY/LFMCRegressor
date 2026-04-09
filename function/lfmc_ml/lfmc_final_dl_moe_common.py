from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lfmc_final_common import VOD_COLS, Y_COL, eval_regression


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


class ResBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResMLPExpert(nn.Module):
    def __init__(self, input_dim, width=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, width)
        self.blocks = nn.Sequential(
            *[ResBlock(width, width * 2, dropout=dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.head(x)


class MoEMLPRegressor(nn.Module):
    def __init__(
        self,
        num_dim,
        n_cats,
        emb_dim=16,
        n_experts=4,
        expert_hidden=(256, 128),
        gate_hidden=128,
        dropout=0.2,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        input_dim = num_dim + emb_dim

        self.experts = nn.ModuleList(
            [MLPExpert(input_dim, hidden=expert_hidden, dropout=dropout) for _ in range(n_experts)]
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, n_experts),
        )

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat[:, 0])], dim=1)
        gate_prob = torch.softmax(self.gate(x), dim=1)

        expert_outs = torch.cat([expert(x) for expert in self.experts], dim=1)
        out = (expert_outs * gate_prob).sum(dim=1, keepdim=True)
        return out


class MoEResNetMLPRegressor(nn.Module):
    def __init__(
        self,
        num_dim,
        n_cats,
        emb_dim=16,
        n_experts=4,
        width=256,
        n_blocks=3,
        gate_hidden=128,
        dropout=0.2,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        input_dim = num_dim + emb_dim

        self.experts = nn.ModuleList(
            [
                ResMLPExpert(
                    input_dim=input_dim,
                    width=width,
                    n_blocks=n_blocks,
                    dropout=dropout,
                )
                for _ in range(n_experts)
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, n_experts),
        )

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat[:, 0])], dim=1)
        gate_prob = torch.softmax(self.gate(x), dim=1)

        expert_outs = torch.cat([expert(x) for expert in self.experts], dim=1)
        out = (expert_outs * gate_prob).sum(dim=1, keepdim=True)
        return out


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


def fit_model(model, train_loader, val_loader, device, max_epochs=120, patience=20):
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


def prepare_dl_category(train_df, val_df, test_df, cat_col):
    all_cats = pd.concat(
        [train_df[cat_col], val_df[cat_col], test_df[cat_col]], axis=0
    ).astype(str).unique()

    cat_map = {v: i for i, v in enumerate(all_cats)}

    out = []
    for df_ in [train_df.copy(), val_df.copy(), test_df.copy()]:
        df_[cat_col] = df_[cat_col].astype(str).map(cat_map).astype(np.int64)
        out.append(df_)

    return out[0], out[1], out[2], len(cat_map)


def prepare_dl_numeric_features(train_df, val_df, test_df, num_cols):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    vod_cols_in_use = [c for c in VOD_COLS if c in num_cols]

    mean = train_df[num_cols].mean()
    std = train_df[num_cols].std().replace(0, 1.0)

    for df_ in [train_df, val_df, test_df]:
        df_[num_cols] = (df_[num_cols] - mean) / std
        if vod_cols_in_use:
            df_[vod_cols_in_use] = df_[vod_cols_in_use].fillna(0.0)

    for split_name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        arr = df_[num_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(arr).all():
            bad_count = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"{split_name} numeric features still contain {bad_count} non-finite values after preprocessing."
            )

    return train_df, val_df, test_df


def make_dataloaders(
    train_df,
    val_df,
    test_df,
    num_cols,
    cat_col,
    batch_size_train=512,
    batch_size_eval=1024,
):
    train_loader = DataLoader(
        LFMCSet(train_df, num_cols, cat_col),
        batch_size=batch_size_train,
        shuffle=True,
    )
    val_loader = DataLoader(
        LFMCSet(val_df, num_cols, cat_col),
        batch_size=batch_size_eval,
        shuffle=False,
    )
    test_loader = DataLoader(
        LFMCSet(test_df, num_cols, cat_col),
        batch_size=batch_size_eval,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
