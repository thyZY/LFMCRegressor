from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "<",
    ">",
    "*",
    "h",
    "8",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot LFMC observed-vs-predicted scatter charts for validation and test splits. "
            "Each subgroup uses a different marker and metrics are shown in the upper-left."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV or parquet file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path, e.g. figs/m1_val_test_scatter.png",
    )
    parser.add_argument(
        "--y-col",
        default="LFMC value (%)",
        help="Observed LFMC column name.",
    )
    parser.add_argument(
        "--pred-col",
        default="lfmc_hat",
        help="Predicted LFMC column name.",
    )
    parser.add_argument(
        "--split-col",
        default="split",
        help="Split column name.",
    )
    parser.add_argument(
        "--group-col",
        default="FT",
        help="Grouping column for marker style, e.g. FT / LC_main / LC_dom.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Splits to plot, default: val test",
    )
    parser.add_argument(
        "--title",
        default="LFMC Observed vs Predicted",
        help="Figure title.",
    )
    parser.add_argument(
        "--max-points-per-group",
        type=int,
        default=2500,
        help="Maximum points sampled per split-group for plotting. Metrics still use full data.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Scatter point alpha.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[13.5, 6.2],
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output dpi.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scatter downsampling.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"n": int(y_true.size), "MAE": mae, "RMSE": rmse, "R2": float(r2)}


def metrics_block(df: pd.DataFrame, y_col: str, pred_col: str, group_col: str) -> str:
    overall = regression_metrics(df[y_col].to_numpy(), df[pred_col].to_numpy())
    lines = [
        "Overall",
        (
            f"n={overall['n']}  "
            f"MAE={overall['MAE']:.2f}  "
            f"RMSE={overall['RMSE']:.2f}  "
            f"R2={overall['R2']:.3f}"
        ),
        "",
        f"By {group_col}",
    ]

    grouped = []
    for group_name, group_df in df.groupby(group_col, dropna=False):
        m = regression_metrics(group_df[y_col].to_numpy(), group_df[pred_col].to_numpy())
        label = "NA" if pd.isna(group_name) else str(group_name)
        grouped.append((label, m["n"], m))

    grouped.sort(key=lambda item: item[1], reverse=True)
    for label, _, m in grouped:
        lines.append(
            f"{label}: n={m['n']}, MAE={m['MAE']:.1f}, RMSE={m['RMSE']:.1f}, R2={m['R2']:.2f}"
        )
    return "\n".join(lines)


def sample_for_plot(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def build_style_maps(groups: list[str]) -> tuple[dict[str, str], dict[str, tuple[float, float, float, float]]]:
    cmap = plt.cm.get_cmap("tab10", max(10, len(groups)))
    marker_map = {group: DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)] for i, group in enumerate(groups)}
    color_map = {group: cmap(i % cmap.N) for i, group in enumerate(groups)}
    return marker_map, color_map


def plot_split(
    ax: plt.Axes,
    df: pd.DataFrame,
    split_name: str,
    y_col: str,
    pred_col: str,
    group_col: str,
    marker_map: dict[str, str],
    color_map: dict[str, tuple[float, float, float, float]],
    max_points: int,
    alpha: float,
    seed: int,
    global_min: float,
    global_max: float,
) -> None:
    plotted_labels = []
    for group_name, group_df in df.groupby(group_col, dropna=False):
        label = "NA" if pd.isna(group_name) else str(group_name)
        plot_df = sample_for_plot(group_df, max_points=max_points, seed=seed)
        ax.scatter(
            plot_df[y_col],
            plot_df[pred_col],
            s=20,
            alpha=alpha,
            marker=marker_map[label],
            color=color_map[label],
            edgecolors="none",
            label=label,
        )
        plotted_labels.append(label)

    ax.plot(
        [global_min, global_max],
        [global_min, global_max],
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.8,
    )
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.set_xlabel("Observed LFMC (%)")
    ax.set_ylabel("Predicted LFMC (%)")
    ax.set_title(f"{split_name} split")
    ax.grid(True, linestyle=":", alpha=0.35)

    metric_text = metrics_block(df, y_col=y_col, pred_col=pred_col, group_col=group_col)
    ax.text(
        0.02,
        0.98,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        bbox={"facecolor": "white", "edgecolor": "#666666", "alpha": 0.90, "boxstyle": "round,pad=0.35"},
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = load_table(input_path)
    require_columns(df, [args.y_col, args.pred_col, args.split_col, args.group_col])

    work = df[[args.y_col, args.pred_col, args.split_col, args.group_col]].copy()
    work[args.y_col] = pd.to_numeric(work[args.y_col], errors="coerce")
    work[args.pred_col] = pd.to_numeric(work[args.pred_col], errors="coerce")
    work = work.dropna(subset=[args.y_col, args.pred_col, args.split_col]).copy()
    work[args.group_col] = work[args.group_col].astype("string").fillna("NA")

    split_frames: list[tuple[str, pd.DataFrame]] = []
    for split_name in args.splits:
        split_df = work.loc[work[args.split_col].astype(str) == split_name].copy()
        if len(split_df) == 0:
            continue
        split_frames.append((split_name, split_df))

    if not split_frames:
        raise ValueError(f"No rows found for requested splits: {args.splits}")

    all_groups = sorted(
        {
            str(group_name)
            for _, split_df in split_frames
            for group_name in split_df[args.group_col].dropna().unique().tolist()
        }
    )
    marker_map, color_map = build_style_maps(all_groups)

    combined = pd.concat([frame for _, frame in split_frames], ignore_index=True)
    global_min = float(np.floor(min(combined[args.y_col].min(), combined[args.pred_col].min()) / 10.0) * 10.0)
    global_max = float(np.ceil(max(combined[args.y_col].max(), combined[args.pred_col].max()) / 10.0) * 10.0)

    fig, axes = plt.subplots(1, len(split_frames), figsize=tuple(args.figsize), constrained_layout=True)
    if len(split_frames) == 1:
        axes = [axes]

    for ax, (split_name, split_df) in zip(axes, split_frames):
        plot_split(
            ax=ax,
            df=split_df,
            split_name=split_name,
            y_col=args.y_col,
            pred_col=args.pred_col,
            group_col=args.group_col,
            marker_map=marker_map,
            color_map=color_map,
            max_points=args.max_points_per_group,
            alpha=args.alpha,
            seed=args.seed,
            global_min=global_min,
            global_max=global_max,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker_map[group],
            color="w",
            markerfacecolor=color_map[group],
            markersize=7,
            linestyle="None",
            label=group,
        )
        for group in all_groups
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(args.title, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
