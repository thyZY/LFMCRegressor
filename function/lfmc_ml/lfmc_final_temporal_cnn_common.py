from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lfmc_final_common import DATE_COL, LAT_COL, LON_COL, VOD_COLS, Y_COL, eval_regression


STATIC_NUM_COLS_DEFAULT = ["Hveg", "lat_norm"]
DYNAMIC_COLS_DEFAULT = VOD_COLS + ["LAI", "LST", "doy_sin", "doy_cos"]
CAT_COL_DEFAULT = "lc_dom"


def _site_key(df: pd.DataFrame) -> pd.Series:
    lat = df[LAT_COL].round(6).astype(str)
    lon = df[LON_COL].round(6).astype(str)
    return lat + "_" + lon


def build_temporal_windows(
    final_df: pd.DataFrame,
    window_size: int,
    static_num_cols: List[str] | None = None,
    dynamic_cols: List[str] | None = None,
    cat_col: str = CAT_COL_DEFAULT,
) -> pd.DataFrame:
    static_num_cols = static_num_cols or STATIC_NUM_COLS_DEFAULT
    dynamic_cols = dynamic_cols or DYNAMIC_COLS_DEFAULT

    need = [DATE_COL, LAT_COL, LON_COL, Y_COL, "split", "sample_weight", cat_col] + static_num_cols + dynamic_cols
    missing = [c for c in need if c not in final_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for temporal windows: {missing}")

    df = final_df[need].copy()
    df["_site_key"] = _site_key(df)

    rows = []
    for _, grp in df.groupby("_site_key", sort=False):
        grp = grp.sort_values(DATE_COL).reset_index(drop=True)
        if len(grp) < window_size:
            continue

        dyn_arr = grp[dynamic_cols].to_numpy(dtype=np.float32)
        for end in range(window_size - 1, len(grp)):
            cur = grp.iloc[end]
            seq = dyn_arr[end - window_size + 1:end + 1].copy()
            rows.append(
                {
                    DATE_COL: cur[DATE_COL],
                    LAT_COL: float(cur[LAT_COL]),
                    LON_COL: float(cur[LON_COL]),
                    "split": cur["split"],
                    "sample_weight": float(cur["sample_weight"]),
                    Y_COL: float(cur[Y_COL]),
                    cat_col: cur[cat_col],
                    "dyn_seq": seq,
                    **{c: float(cur[c]) for c in static_num_cols},
                }
            )

    return pd.DataFrame(rows)


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
        return values.astype(str).map(lambda x: self.mapping.get(x, self.unk_id)).to_numpy(dtype=np.int64)

    def to_dict(self) -> Dict:
        return {"mapping": self.mapping, "unk_id": self.unk_id}

    @classmethod
    def from_dict(cls, d: Dict) -> "LabelEncoderWithUNK":
        obj = cls()
        obj.mapping = {str(k): int(v) for k, v in d["mapping"].items()}
        obj.unk_id = int(d["unk_id"])
        return obj


def prepare_temporal_category(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, LabelEncoderWithUNK]:
    enc = LabelEncoderWithUNK().fit(train_df[cat_col])
    out = []
    for df_ in [train_df.copy(), val_df.copy(), test_df.copy()]:
        df_[cat_col] = enc.transform(df_[cat_col])
        out.append(df_)
    return out[0], out[1], out[2], len(enc.mapping) + 1, enc


def _stack_dyn(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["dyn_seq"].to_list(), axis=0).astype(np.float32)


def prepare_temporal_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    static_num_cols: List[str],
    dynamic_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    static_mean = train_df[static_num_cols].mean()
    static_std = train_df[static_num_cols].std().replace(0, 1.0)

    train_dyn = _stack_dyn(train_df)
    dyn_mean = np.nanmean(train_dyn, axis=(0, 1)).astype(np.float32)
    dyn_std = np.nanstd(train_dyn, axis=(0, 1)).astype(np.float32)
    dyn_std[dyn_std == 0] = 1.0

    vod_idx = [dynamic_cols.index(c) for c in dynamic_cols if c in VOD_COLS]

    def _transform(df_: pd.DataFrame):
        out = df_.copy()
        out[static_num_cols] = (out[static_num_cols] - static_mean) / static_std

        seqs = []
        for seq in out["dyn_seq"].to_list():
            seq = np.asarray(seq, dtype=np.float32)
            seq = (seq - dyn_mean.reshape(1, -1)) / dyn_std.reshape(1, -1)
            if vod_idx:
                seq[:, vod_idx] = np.where(np.isnan(seq[:, vod_idx]), 0.0, seq[:, vod_idx])
            seq = np.where(np.isfinite(seq), seq, np.nan)
            seqs.append(seq.astype(np.float32))
        out["dyn_seq"] = seqs
        return out

    train_df = _transform(train_df)
    val_df = _transform(val_df)
    test_df = _transform(test_df)

    for split_name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        static_arr = df_[static_num_cols].to_numpy(dtype=np.float32)
        dyn_arr = _stack_dyn(df_)
        bad_static = int((~np.isfinite(static_arr)).sum())
        bad_dyn = int((~np.isfinite(dyn_arr)).sum())
        if bad_static > 0:
            raise ValueError(f"{split_name} static features still contain {bad_static} non-finite values.")
        if bad_dyn > 0:
            raise ValueError(f"{split_name} dynamic features still contain {bad_dyn} non-finite values.")

    stats = {
        "static_mean": static_mean.to_dict(),
        "static_std": static_std.to_dict(),
        "dynamic_mean": dyn_mean.tolist(),
        "dynamic_std": dyn_std.tolist(),
    }
    return train_df, val_df, test_df, stats


def transform_temporal_numeric_features(
    df: pd.DataFrame,
    static_num_cols: List[str],
    dynamic_cols: List[str],
    stats: Dict,
) -> pd.DataFrame:
    out = df.copy()
    static_mean = pd.Series(stats["static_mean"])
    static_std = pd.Series(stats["static_std"]).replace(0, 1.0)
    dyn_mean = np.asarray(stats["dynamic_mean"], dtype=np.float32)
    dyn_std = np.asarray(stats["dynamic_std"], dtype=np.float32)
    dyn_std[dyn_std == 0] = 1.0

    out[static_num_cols] = (out[static_num_cols] - static_mean[static_num_cols]) / static_std[static_num_cols]

    vod_idx = [dynamic_cols.index(c) for c in dynamic_cols if c in VOD_COLS]
    seqs = []
    for seq in out["dyn_seq"].to_list():
        seq = np.asarray(seq, dtype=np.float32)
        seq = (seq - dyn_mean.reshape(1, -1)) / dyn_std.reshape(1, -1)
        if vod_idx:
            seq[:, vod_idx] = np.where(np.isnan(seq[:, vod_idx]), 0.0, seq[:, vod_idx])
        seq = np.where(np.isfinite(seq), seq, np.nan)
        seqs.append(seq.astype(np.float32))
    out["dyn_seq"] = seqs
    return out


class TemporalLFMCSet(Dataset):
    def __init__(self, df: pd.DataFrame, static_num_cols: List[str], cat_col: str):
        self.x_static = torch.tensor(df[static_num_cols].to_numpy(np.float32), dtype=torch.float32)
        self.x_cat = torch.tensor(df[[cat_col]].to_numpy(np.int64), dtype=torch.long)
        self.x_dyn = torch.tensor(np.stack(df["dyn_seq"].to_list(), axis=0), dtype=torch.float32)
        self.y = torch.tensor(df[Y_COL].to_numpy(np.float32), dtype=torch.float32)
        self.w = torch.tensor(df["sample_weight"].to_numpy(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_static[idx], self.x_cat[idx], self.x_dyn[idx], self.y[idx], self.w[idx]


class StaticTower(nn.Module):
    def __init__(self, static_dim: int, n_cats: int, emb_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(static_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x_static, x_cat):
        return self.net(torch.cat([x_static, self.emb(x_cat[:, 0])], dim=1))


class ResBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.2):
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


class ResNetStaticTower(nn.Module):
    def __init__(self, static_dim: int, n_cats: int, emb_dim: int = 8, width: int = 64, n_blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        self.in_proj = nn.Linear(static_dim + emb_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width, width * 2, dropout=dropout) for _ in range(n_blocks)])
        self.out_proj = nn.Sequential(nn.LayerNorm(width), nn.ReLU())

    def forward(self, x_static, x_cat):
        x = torch.cat([x_static, self.emb(x_cat[:, 0])], dim=1)
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out_proj(x)


class GLULayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class TabNetStaticTower(nn.Module):
    def __init__(self, static_dim: int, n_cats: int, emb_dim: int = 8, hidden_dim: int = 64, n_steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        self.input_dim = static_dim + emb_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps

        self.bn = nn.BatchNorm1d(self.input_dim)
        self.initial_proj = nn.Linear(self.input_dim, hidden_dim)
        self.initial = nn.Sequential(GLULayer(hidden_dim, hidden_dim), nn.Dropout(dropout), GLULayer(hidden_dim, hidden_dim))
        self.masked_proj = nn.Linear(self.input_dim, hidden_dim)
        self.step_blocks = nn.ModuleList(
            [nn.Sequential(GLULayer(hidden_dim, hidden_dim), nn.Dropout(dropout), GLULayer(hidden_dim, hidden_dim)) for _ in range(n_steps)]
        )
        self.mask_blocks = nn.ModuleList([nn.Linear(hidden_dim, self.input_dim) for _ in range(n_steps)])

    def forward(self, x_static, x_cat):
        x = torch.cat([x_static, self.emb(x_cat[:, 0])], dim=1)
        x = self.bn(x)
        prior = torch.ones_like(x)
        h = self.initial(self.initial_proj(x))
        decisions = []
        for step in range(self.n_steps):
            mask = torch.softmax(self.mask_blocks[step](h), dim=1)
            masked_h = self.masked_proj(x * mask * prior)
            h = self.step_blocks[step](h + masked_h)
            decisions.append(h)
            prior = prior * (1.5 - mask)
        return torch.stack(decisions, dim=0).mean(dim=0)


class TemporalConvTower(nn.Module):
    def __init__(self, dynamic_dim: int, channels: int = 64, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(dynamic_dim, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_dyn):
        x = x_dyn.transpose(1, 2)
        x = self.net(x)
        return self.pool(x).squeeze(-1)


class DualBranchTemporalCNNRegressor(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        n_cats: int,
        emb_dim: int = 8,
        static_hidden: int = 64,
        temporal_channels: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.static_tower = StaticTower(
            static_dim=static_dim,
            n_cats=n_cats,
            emb_dim=emb_dim,
            hidden_dim=static_hidden,
            dropout=dropout,
        )
        self.temporal_tower = TemporalConvTower(
            dynamic_dim=dynamic_dim,
            channels=temporal_channels,
            kernel_size=3,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(static_hidden + temporal_channels, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_static, x_cat, x_dyn):
        x_static_ctx = self.static_tower(x_static, x_cat)
        x_temporal_ctx = self.temporal_tower(x_dyn)
        return self.head(torch.cat([x_static_ctx, x_temporal_ctx], dim=1))


class DualBranchTemporalCNNResNetRegressor(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        n_cats: int,
        emb_dim: int = 8,
        static_width: int = 64,
        static_blocks: int = 2,
        temporal_channels: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.static_tower = ResNetStaticTower(
            static_dim=static_dim,
            n_cats=n_cats,
            emb_dim=emb_dim,
            width=static_width,
            n_blocks=static_blocks,
            dropout=dropout,
        )
        self.temporal_tower = TemporalConvTower(dynamic_dim=dynamic_dim, channels=temporal_channels, kernel_size=3, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(static_width + temporal_channels, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_static, x_cat, x_dyn):
        x_static_ctx = self.static_tower(x_static, x_cat)
        x_temporal_ctx = self.temporal_tower(x_dyn)
        return self.head(torch.cat([x_static_ctx, x_temporal_ctx], dim=1))


class DualBranchTemporalCNNTabNetRegressor(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        n_cats: int,
        emb_dim: int = 8,
        static_hidden: int = 64,
        temporal_channels: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.static_tower = TabNetStaticTower(
            static_dim=static_dim,
            n_cats=n_cats,
            emb_dim=emb_dim,
            hidden_dim=static_hidden,
            n_steps=2,
            dropout=dropout,
        )
        self.temporal_tower = TemporalConvTower(dynamic_dim=dynamic_dim, channels=temporal_channels, kernel_size=3, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(static_hidden + temporal_channels, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_static, x_cat, x_dyn):
        x_static_ctx = self.static_tower(x_static, x_cat)
        x_temporal_ctx = self.temporal_tower(x_dyn)
        return self.head(torch.cat([x_static_ctx, x_temporal_ctx], dim=1))


class MoEFusionHead(nn.Module):
    def __init__(self, input_dim: int, n_experts: int = 4, expert_hidden: int = 128, gate_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, 1),
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

    def forward(self, x):
        gate_prob = torch.softmax(self.gate(x), dim=1)
        expert_outs = torch.cat([expert(x) for expert in self.experts], dim=1)
        return (expert_outs * gate_prob).sum(dim=1, keepdim=True)


class DualBranchTemporalCNNMoERegressor(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        n_cats: int,
        emb_dim: int = 8,
        static_hidden: int = 64,
        temporal_channels: int = 64,
        n_experts: int = 4,
        expert_hidden: int = 128,
        gate_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.static_tower = StaticTower(static_dim=static_dim, n_cats=n_cats, emb_dim=emb_dim, hidden_dim=static_hidden, dropout=dropout)
        self.temporal_tower = TemporalConvTower(dynamic_dim=dynamic_dim, channels=temporal_channels, kernel_size=3, dropout=dropout)
        self.fusion = MoEFusionHead(
            input_dim=static_hidden + temporal_channels,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            gate_hidden=gate_hidden,
            dropout=dropout,
        )

    def forward(self, x_static, x_cat, x_dyn):
        x_static_ctx = self.static_tower(x_static, x_cat)
        x_temporal_ctx = self.temporal_tower(x_dyn)
        return self.fusion(torch.cat([x_static_ctx, x_temporal_ctx], dim=1))


class DualBranchTemporalCNNResNetMoERegressor(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        n_cats: int,
        emb_dim: int = 8,
        static_width: int = 64,
        static_blocks: int = 2,
        temporal_channels: int = 64,
        n_experts: int = 4,
        expert_hidden: int = 128,
        gate_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.static_tower = ResNetStaticTower(
            static_dim=static_dim,
            n_cats=n_cats,
            emb_dim=emb_dim,
            width=static_width,
            n_blocks=static_blocks,
            dropout=dropout,
        )
        self.temporal_tower = TemporalConvTower(dynamic_dim=dynamic_dim, channels=temporal_channels, kernel_size=3, dropout=dropout)
        self.fusion = MoEFusionHead(
            input_dim=static_width + temporal_channels,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            gate_hidden=gate_hidden,
            dropout=dropout,
        )

    def forward(self, x_static, x_cat, x_dyn):
        x_static_ctx = self.static_tower(x_static, x_cat)
        x_temporal_ctx = self.temporal_tower(x_dyn)
        return self.fusion(torch.cat([x_static_ctx, x_temporal_ctx], dim=1))


class DualBranchTemporalCNNTabNetMoERegressor(nn.Module):
    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        n_cats: int,
        emb_dim: int = 8,
        static_hidden: int = 64,
        temporal_channels: int = 64,
        n_experts: int = 4,
        expert_hidden: int = 128,
        gate_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.static_tower = TabNetStaticTower(
            static_dim=static_dim,
            n_cats=n_cats,
            emb_dim=emb_dim,
            hidden_dim=static_hidden,
            n_steps=2,
            dropout=dropout,
        )
        self.temporal_tower = TemporalConvTower(dynamic_dim=dynamic_dim, channels=temporal_channels, kernel_size=3, dropout=dropout)
        self.fusion = MoEFusionHead(
            input_dim=static_hidden + temporal_channels,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            gate_hidden=gate_hidden,
            dropout=dropout,
        )

    def forward(self, x_static, x_cat, x_dyn):
        x_static_ctx = self.static_tower(x_static, x_cat)
        x_temporal_ctx = self.temporal_tower(x_dyn)
        return self.fusion(torch.cat([x_static_ctx, x_temporal_ctx], dim=1))


def make_temporal_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    static_num_cols: List[str],
    cat_col: str,
    batch_size_train: int = 256,
    batch_size_eval: int = 512,
):
    train_loader = DataLoader(
        TemporalLFMCSet(train_df, static_num_cols, cat_col),
        batch_size=batch_size_train,
        shuffle=True,
    )
    val_loader = DataLoader(
        TemporalLFMCSet(val_df, static_num_cols, cat_col),
        batch_size=batch_size_eval,
        shuffle=False,
    )
    test_loader = DataLoader(
        TemporalLFMCSet(test_df, static_num_cols, cat_col),
        batch_size=batch_size_eval,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


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

    for x_static, x_cat, x_dyn, y, w in loader:
        x_static = x_static.to(device)
        x_cat = x_cat.to(device)
        x_dyn = x_dyn.to(device)
        y = y.to(device)
        w = w.to(device)

        pred = model(x_static, x_cat, x_dyn)
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


def fit_model(model, train_loader, val_loader, device, max_epochs: int = 80, patience: int = 12):
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_temporal_cnn_artifact(
    artifact_dir: Path,
    model: nn.Module,
    model_name: str,
    model_kwargs: Dict,
    window_size: int,
    static_num_cols: List[str],
    dynamic_cols: List[str],
    cat_col: str,
    stats: Dict,
    encoder: LabelEncoderWithUNK,
):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "model_kwargs": model_kwargs,
        },
        artifact_dir / "model.pt",
    )
    save_json(
        {
            "window_size": window_size,
            "static_num_cols": static_num_cols,
            "dynamic_cols": dynamic_cols,
            "cat_col": cat_col,
            "stats": stats,
            "encoder": encoder.to_dict(),
        },
        artifact_dir / "preprocess.json",
    )
