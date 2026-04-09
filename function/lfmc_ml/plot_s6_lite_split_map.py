from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from lfmc_final_common import (
    DATA_PATH,
    OUT_DIR,
    RANDOM_SEED,
    SPLIT_BLOCK_DEG,
    build_final_s6_lite_frame,
    load_base_dataframe,
)


DEFAULT_OUT_DIR = OUT_DIR / "split_maps"

SPLIT_TO_CODE = {
    "train": 1,
    "val": 2,
    "test": 3,
}

SPLIT_TO_LABEL = {
    0: "No data",
    1: "Train",
    2: "Validation",
    3: "Test",
}

SPLIT_TO_COLOR = {
    0: "#f2f2f2",
    1: "#4c78a8",
    2: "#f58518",
    3: "#54a24b",
}


def build_block_table(final_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["block_id", "_block_lat", "_block_lon", "split"]
    missing = [c for c in cols if c not in final_df.columns]
    if missing:
        raise ValueError(f"Missing block columns in final_df: {missing}")

    block_df = (
        final_df[cols]
        .drop_duplicates(subset=["block_id"])
        .copy()
        .sort_values(["_block_lat", "_block_lon"])
        .reset_index(drop=True)
    )

    block_df["lat_min"] = -90.0 + block_df["_block_lat"] * SPLIT_BLOCK_DEG
    block_df["lat_max"] = block_df["lat_min"] + SPLIT_BLOCK_DEG
    block_df["lon_min"] = -180.0 + block_df["_block_lon"] * SPLIT_BLOCK_DEG
    block_df["lon_max"] = block_df["lon_min"] + SPLIT_BLOCK_DEG
    block_df["lat_center"] = (block_df["lat_min"] + block_df["lat_max"]) / 2.0
    block_df["lon_center"] = (block_df["lon_min"] + block_df["lon_max"]) / 2.0
    block_df["split_code"] = block_df["split"].map(SPLIT_TO_CODE).astype(np.int16)
    return block_df


def build_split_grid(block_df: pd.DataFrame) -> np.ndarray:
    n_lat = int(round(180.0 / SPLIT_BLOCK_DEG))
    n_lon = int(round(360.0 / SPLIT_BLOCK_DEG))
    grid = np.zeros((n_lat, n_lon), dtype=np.int16)

    lat_idx = block_df["_block_lat"].to_numpy(dtype=np.int32)
    lon_idx = block_df["_block_lon"].to_numpy(dtype=np.int32)
    split_code = block_df["split_code"].to_numpy(dtype=np.int16)
    grid[lat_idx, lon_idx] = split_code
    return grid


def save_block_map(block_df: pd.DataFrame, out_path: Path, title: str):
    grid = build_split_grid(block_df)
    cmap = ListedColormap([SPLIT_TO_COLOR[i] for i in range(4)])

    fig, ax = plt.subplots(figsize=(18, 9), dpi=200)
    ax.imshow(
        grid,
        origin="lower",
        extent=(-180.0, 180.0, -90.0, 90.0),
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=3,
        aspect="auto",
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(color="#d9d9d9", linewidth=0.3, alpha=0.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SPLIT_TO_COLOR[1], label="Train"),
        plt.Rectangle((0, 0), 1, 1, color=SPLIT_TO_COLOR[2], label="Validation"),
        plt.Rectangle((0, 0), 1, 1, color=SPLIT_TO_COLOR[3], label="Test"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_train_val_only_map(block_df: pd.DataFrame, out_path: Path, title: str):
    tv_df = block_df[block_df["split"].isin(["train", "val"])].copy()
    grid = build_split_grid(tv_df)
    cmap = ListedColormap([SPLIT_TO_COLOR[0], SPLIT_TO_COLOR[1], SPLIT_TO_COLOR[2], SPLIT_TO_COLOR[0]])

    fig, ax = plt.subplots(figsize=(18, 9), dpi=200)
    ax.imshow(
        grid,
        origin="lower",
        extent=(-180.0, 180.0, -90.0, 90.0),
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=3,
        aspect="auto",
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(color="#d9d9d9", linewidth=0.3, alpha=0.5)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SPLIT_TO_COLOR[1], label="Train"),
        plt.Rectangle((0, 0), 1, 1, color=SPLIT_TO_COLOR[2], label="Validation"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_base_dataframe(Path(args.data_path))
    final_df, _, _ = build_final_s6_lite_frame(base_df)
    block_df = build_block_table(final_df)

    block_csv = out_dir / "s6_lite_split_blocks.csv"
    block_df.to_csv(block_csv, index=False, encoding="utf-8-sig")

    save_block_map(
        block_df,
        out_dir / "s6_lite_split_map_all.png",
        title=f"S6-Lite Spatial Blocks ({SPLIT_BLOCK_DEG}° x {SPLIT_BLOCK_DEG}°, seed={RANDOM_SEED})",
    )
    save_train_val_only_map(
        block_df,
        out_dir / "s6_lite_split_map_train_val.png",
        title=f"S6-Lite Train / Validation Blocks ({SPLIT_BLOCK_DEG}° x {SPLIT_BLOCK_DEG}°, seed={RANDOM_SEED})",
    )

    summary = block_df["split"].value_counts().rename_axis("split").reset_index(name="n_blocks")
    summary.to_csv(out_dir / "s6_lite_split_block_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Saved block table to: {block_csv}")
    print(f"Saved map to: {out_dir / 's6_lite_split_map_all.png'}")
    print(f"Saved train/val map to: {out_dir / 's6_lite_split_map_train_val.png'}")
    print(f"Saved summary to: {out_dir / 's6_lite_split_block_summary.csv'}")


if __name__ == "__main__":
    main()
