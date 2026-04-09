from __future__ import annotations

import copy
import random
from typing import Tuple

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


class ResNetMLPRegressor(nn.Module):
    def __init__(
        self,
        num_dim,
        n_cats,
        emb_dim=16,
        width=256,
        n_blocks=3,
        dropout=0.2,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        self.in_proj = nn.Linear(num_dim + emb_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width, width * 2, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat[:, 0])], dim=1)
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.head(x)


class TabTransformerBase(nn.Module):
    def __init__(
        self,
        num_dim,
        n_cats,
        emb_dim=16,
        n_heads=4,
        n_layers=2,
        hidden=(256, 128),
        dropout=0.2,
    ):
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


class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerRegressor(nn.Module):
    def __init__(
        self,
        num_dim,
        n_cats,
        d_token=32,
        n_heads=4,
        n_layers=3,
        dropout=0.2,
        ff_mult=4,
    ):
        super().__init__()
        self.num_tok = NumericalFeatureTokenizer(num_dim, d_token)
        self.cat_emb = nn.Embedding(n_cats, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_num, x_cat):
        num_tokens = self.num_tok(x_num)
        cat_token = self.cat_emb(x_cat[:, 0]).unsqueeze(1)
        cls = self.cls.expand(x_num.size(0), -1, -1)
        x = torch.cat([cls, cat_token, num_tokens], dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0])


class GLULayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class SimpleTabNetRegressor(nn.Module):
    """
    Lightweight TabNet-style regressor.
    Not a full reference implementation, but keeps the basic idea:
    sequential feature masking + decision aggregation.

    Important fix:
    masked_x lives in input space, while h lives in hidden space.
    So masked_x must be projected before being added to h.
    """
    def __init__(
        self,
        num_dim,
        n_cats,
        emb_dim=8,
        hidden_dim=128,
        decision_dim=64,
        n_steps=3,
        gamma=1.5,
        dropout=0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        self.input_dim = num_dim + emb_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.gamma = gamma

        self.bn = nn.BatchNorm1d(self.input_dim)

        self.initial_proj = nn.Linear(self.input_dim, hidden_dim)
        self.initial = nn.Sequential(
            GLULayer(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            GLULayer(hidden_dim, hidden_dim),
        )

        self.masked_proj = nn.Linear(self.input_dim, hidden_dim)

        self.step_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    GLULayer(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                    GLULayer(hidden_dim, hidden_dim),
                )
                for _ in range(n_steps)
            ]
        )
        self.mask_blocks = nn.ModuleList(
            [nn.Linear(hidden_dim, self.input_dim) for _ in range(n_steps)]
        )
        self.decision_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, decision_dim) for _ in range(n_steps)]
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(decision_dim, 1),
        )

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat[:, 0])], dim=1)
        x = self.bn(x)

        prior = torch.ones_like(x)

        h = self.initial(self.initial_proj(x))
        decisions = []

        for step in range(self.n_steps):
            mask_logits = self.mask_blocks[step](h)
            mask = torch.softmax(mask_logits, dim=1)

            masked_x = x * mask * prior
            masked_h = self.masked_proj(masked_x)

            h = self.step_blocks[step](h + masked_h)
            decisions.append(self.decision_proj[step](h))

            prior = prior * (self.gamma - mask)

        agg = torch.stack(decisions, dim=0).sum(dim=0)
        return self.head(agg)


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


def fit_model(model, train_loader, val_loader, device, max_epochs=100, patience=15):
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


def prepare_dl_category(train_df, val_df, test_df, cat_col) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
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
