from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

def parse_math_heatmap_args():
    parser = argparse.ArgumentParser(
        description="Render heatmaps from pairwise mathematical comparison metrics."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to pairwise_method_metrics.csv produced by compare_math_results.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_math_heatmaps",
        help="Directory for generated heatmaps and matrix CSV files.",
    )
    parser.add_argument(
        "--title",
        default="Mathematical Comparison Heatmap",
        help="Base title used in plots.",
    )
    return parser.parse_args()


def collect_items(df: pd.DataFrame) -> list[str]:
    items = sorted(set(df["item_a"]).union(df["item_b"]))
    return items


def display_label(item: str) -> str:
    label = item.replace("_", " ")
    return label[:1].upper() + label[1:] if label else label


def build_symmetric_matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    items = collect_items(df)
    matrix = pd.DataFrame(np.nan, index=items, columns=items, dtype=float)
    np.fill_diagonal(matrix.values, 1.0)

    for row in df.itertuples(index=False):
        value = getattr(row, metric)
        matrix.loc[row.item_a, row.item_b] = value
        matrix.loc[row.item_b, row.item_a] = value
    return matrix


def plot_matrix(matrix: pd.DataFrame, metric: str, output_dir: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)

    if metric == "spearman_relative_error":
        vmin, vmax = -1.0, 1.0
        cmap = "coolwarm"
        cbar_label = "Spearman correlation"
    else:
        vmin, vmax = 0.0, 1.0
        cmap = "viridis"
        cbar_label = metric.replace("_", " ")

    display = matrix.values.astype(float)
    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels([display_label(item) for item in matrix.columns], rotation=45, ha="right")
    ax.set_yticklabels([display_label(item) for item in matrix.index])
    #ax.set_title(f"{title}: {metric}")

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display[i, j]
            if np.isnan(value):
                text = "NA"
            else:
                text = f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=12, color="black")

    png_path = output_dir / f"{metric}_heatmap.png"
    pdf_path = output_dir / f"{metric}_heatmap.pdf"
    fig.savefig(png_path, dpi=170)
    fig.savefig(pdf_path)
    plt.close(fig)


def run_math_heatmap(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"item_a", "item_b", "spearman_relative_error", "event_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {sorted(missing)}")

    for metric in ["spearman_relative_error", "event_f1", "jaccard_anomaly_hours"]:
        if metric not in df.columns:
            continue
        matrix = build_symmetric_matrix(df, metric)
        matrix.to_csv(output_dir / f"{metric}_matrix.csv")
        plot_matrix(matrix, metric, output_dir, args.title)

    print("\nDone.")
    print(f"Saved heatmaps to {output_dir}")
