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

from common_features import TARGET_COL, VOD_FEATURES


Y_COL = TARGET_COL
VOD_COLS = [item[0] for item in VOD_FEATURES]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


class ResNetMLPRegressor(nn.Module):
    def __init__(self, num_dim, n_cats, emb_dim=16, width=256, n_blocks=3, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        self.in_proj = nn.Linear(num_dim + emb_dim, width)
        self.blocks = nn.Sequential(*[ResBlock(width, width * 2, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.emb(x_cat[:, 0])], dim=1)
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.head(x)


class GLULayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class SimpleTabNetRegressor(nn.Module):
    def __init__(self, num_dim, n_cats, emb_dim=8, hidden_dim=128, decision_dim=64, n_steps=3, gamma=1.5, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        self.input_dim = num_dim + emb_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.gamma = gamma

        self.bn = nn.BatchNorm1d(self.input_dim)
        self.initial_proj = nn.Linear(self.input_dim, hidden_dim)
        self.initial = nn.Sequential(GLULayer(hidden_dim, hidden_dim), nn.Dropout(dropout), GLULayer(hidden_dim, hidden_dim))
        self.masked_proj = nn.Linear(self.input_dim, hidden_dim)
        self.step_blocks = nn.ModuleList([
            nn.Sequential(GLULayer(hidden_dim, hidden_dim), nn.Dropout(dropout), GLULayer(hidden_dim, hidden_dim))
            for _ in range(n_steps)
        ])
        self.mask_blocks = nn.ModuleList([nn.Linear(hidden_dim, self.input_dim) for _ in range(n_steps)])
        self.decision_proj = nn.ModuleList([nn.Linear(hidden_dim, decision_dim) for _ in range(n_steps)])
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(decision_dim, 1))

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
        self.blocks = nn.Sequential(*[ResBlock(width, width * 2, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.head(x)


class MoEResNetMLPRegressor(nn.Module):
    def __init__(self, num_dim, n_cats, emb_dim=16, n_experts=4, width=256, n_blocks=3, gate_hidden=128, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb_dim)
        input_dim = num_dim + emb_dim
        self.experts = nn.ModuleList([
            ResMLPExpert(input_dim=input_dim, width=width, n_blocks=n_blocks, dropout=dropout)
            for _ in range(n_experts)
        ])
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


MODEL_REGISTRY = {
    "ResNetMLP": ResNetMLPRegressor,
    "TabNet": SimpleTabNetRegressor,
    "MoE_ResNetMLP": MoEResNetMLPRegressor,
}

MODEL_DEFAULT_KWARGS = {
    "ResNetMLP": {"emb_dim": 16, "width": 256, "n_blocks": 3, "dropout": 0.2},
    "TabNet": {"emb_dim": 8, "hidden_dim": 128, "decision_dim": 64, "n_steps": 3, "gamma": 1.5, "dropout": 0.1},
    "MoE_ResNetMLP": {"emb_dim": 16, "n_experts": 4, "width": 256, "n_blocks": 3, "gate_hidden": 128, "dropout": 0.2},
}


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


def prepare_dl_category(train_df, val_df, test_df, cat_col):
    enc = LabelEncoderWithUNK().fit(train_df[cat_col])
    out = []
    for df_ in [train_df.copy(), val_df.copy(), test_df.copy()]:
        df_[cat_col] = enc.transform(df_[cat_col])
        out.append(df_)
    return out[0], out[1], out[2], len(enc.mapping) + 1, enc


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
            raise ValueError(f"{split_name} numeric features still contain {bad_count} non-finite values after preprocessing.")

    stats = {"mean": mean.to_dict(), "std": std.to_dict()}
    return train_df, val_df, test_df, stats


def transform_numeric_features(df: pd.DataFrame, num_cols: List[str], stats: Dict):
    out = df.copy()
    mean = pd.Series(stats["mean"])
    std = pd.Series(stats["std"]).replace(0, 1.0)
    vod_cols_in_use = [c for c in VOD_COLS if c in num_cols]
    out[num_cols] = (out[num_cols] - mean[num_cols]) / std[num_cols]
    if vod_cols_in_use:
        out[vod_cols_in_use] = out[vod_cols_in_use].fillna(0.0)
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    if out[num_cols].isna().any().any():
        raise ValueError("Inference numeric features contain NaN after preprocessing.")
    return out


def make_dataloaders(train_df, val_df, test_df, num_cols, cat_col, batch_size_train=512, batch_size_eval=1024):
    train_loader = DataLoader(LFMCSet(train_df, num_cols, cat_col), batch_size=batch_size_train, shuffle=True)
    val_loader = DataLoader(LFMCSet(val_df, num_cols, cat_col), batch_size=batch_size_eval, shuffle=False)
    test_loader = DataLoader(LFMCSet(test_df, num_cols, cat_col), batch_size=batch_size_eval, shuffle=False)
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
    mae = np.mean(np.abs(preds_all - y_all))
    rmse = np.sqrt(np.mean((preds_all - y_all) ** 2))
    r = np.corrcoef(y_all, preds_all)[0, 1] if len(y_all) >= 2 else np.nan
    return total_loss / max(n, 1), {"MAE": float(mae), "RMSE": float(rmse), "R": float(r)}


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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_dl_artifact(artifact_dir: Path, model_name: str, model: nn.Module, model_kwargs: Dict, num_cols: List[str], cat_col: str, stats: Dict, encoder: LabelEncoderWithUNK):
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
            "num_cols": num_cols,
            "cat_col": cat_col,
            "stats": stats,
            "encoder": encoder.to_dict(),
        },
        artifact_dir / "preprocess.json",
    )


def load_dl_artifact(artifact_dir: Path, device: torch.device):
    ckpt = torch.load(artifact_dir / "model.pt", map_location=device)
    cfg = load_json(artifact_dir / "preprocess.json")
    model_name = ckpt["model_name"]
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    encoder = LabelEncoderWithUNK.from_dict(cfg["encoder"])
    return model_name, model, cfg["num_cols"], cfg["cat_col"], cfg["stats"], encoder
