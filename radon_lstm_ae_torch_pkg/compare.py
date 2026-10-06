import argparse
import colorsys
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import dates as mdates
from matplotlib.colors import Normalize, to_rgb


BASE_RESULT_COLUMNS = {"timestamp", "reconstruction_error", "threshold", "relative_error", "is_anomaly"}


@dataclass
class ComparisonSeries:
    label: str
    display_label: str
    df: pd.DataFrame
    radon_col: str | None
    norm_score_col: str
    norm_score_label: str


def parse_compare_args():
    parser = argparse.ArgumentParser(
        description="Compare anomaly result CSVs from different methods or stations."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Result CSV files, typically sequence_scores.csv outputs.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels for the inputs. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_results",
        help="Directory for comparison plots and merged CSV output.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column name in the result CSVs.",
    )
    parser.add_argument(
        "--score-col",
        default="reconstruction_error",
        help="Score column name in the result CSVs.",
    )
    parser.add_argument(
        "--anomaly-col",
        default="is_anomaly",
        help="Anomaly-flag column name in the result CSVs.",
    )
    parser.add_argument(
        "--normalization-percentile",
        type=float,
        default=95.0,
        help="Percentile used to scale each score series for normalized comparison.",
    )
    parser.add_argument(
        "--title",
        default="Anomaly Score Comparison",
        help="Base title used in generated plots.",
    )
    parser.add_argument(
        "--plot-fontsize",
        type=float,
        default=11.0,
        help="Base font size used in comparison plots.",
    )
    parser.add_argument(
        "--plot-width",
        type=float,
        default=14.0,
        help="Plot width in inches for comparison figures.",
    )
    parser.add_argument(
        "--plot-height",
        type=float,
        default=4.2,
        help="Plot height in inches for the score figures.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=170,
        help="DPI used when saving comparison plots.",
    )
    parser.add_argument(
        "--raster-row-height",
        type=float,
        default=0.42,
        help="Vertical space in inches allocated to each series row in the anomaly-dot plot.",
    )
    parser.add_argument(
        "--raster-dot-size",
        type=float,
        default=20.0,
        help="Dot size used in the anomaly-dot plot.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional inclusive start date/time for plotting, for example 2022-11-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional inclusive end date/time for plotting, for example 2026-03-31.",
    )
    parser.add_argument(
        "--month-tick-interval",
        type=int,
        default=4,
        help="Month interval used for x-axis ticks.",
    )
    parser.add_argument(
        "--match-redraw-plot-box",
        action="store_true",
        help=(
            "Expand the anomaly comparison canvas so its actual plotting box matches the "
            "redraw_result_plots.py layout for the same --plot-width."
        ),
    )
    return parser.parse_args()


def plot_rc_params(args) -> dict[str, float]:
    base = float(getattr(args, "plot_fontsize", 11.0))
    return {
        "font.size": base,
        "axes.titlesize": base + 3,
        "axes.labelsize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
    }


def save_figure(fig, output_path: str, args):
    fig.savefig(output_path, dpi=int(getattr(args, "plot_dpi", 170)))


def style_time_axis(ax, args):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, int(args.month_tick_interval))))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax.tick_params(axis="x", rotation=0)


def apply_fixed_margins(fig, left: float = 0.075, right: float = 0.99, bottom: float = 0.14, top: float = 0.90):
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def adjusted_figure_width_for_plot_box(base_width: float, left: float, right: float, args) -> float:
    if not getattr(args, "match_redraw_plot_box", False):
        return base_width
    # Match the redraw_result_plots.py axes-box width:
    # reference box = base_width * (0.99 - 0.075)
    reference_fraction = 0.99 - 0.075
    current_fraction = right - left
    if current_fraction <= 0:
        return base_width
    return base_width * (reference_fraction / current_fraction)


def get_plot_range(args):
    # Callers like math_compare.py reuse load_result_series without defining
    # the plot-range arguments, so treat them as optional.
    start = None
    end = None
    start_date = getattr(args, "start_date", None)
    end_date = getattr(args, "end_date", None)
    if start_date:
        start = pd.to_datetime(start_date, utc=True, errors="coerce")
        if pd.isna(start):
            raise ValueError(f"Invalid --start-date: {start_date}")
    if end_date:
        end = pd.to_datetime(end_date, utc=True, errors="coerce")
        if pd.isna(end):
            raise ValueError(f"Invalid --end-date: {end_date}")
    return start, end


def apply_plot_range(ax, args):
    start, end = get_plot_range(args)
    if start is not None or end is not None:
        ax.set_xlim(left=start, right=end)


def default_labels(paths: list[str]) -> list[str]:
    return [os.path.splitext(os.path.basename(path))[0] for path in paths]


def to_display_label(label: str) -> str:
    parts = [part for part in label.replace("-", "_").split("_") if part]
    if len(parts) <= 1:
        return label
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    split_idx = len(parts) // 2
    return f"{' '.join(parts[:split_idx])}\n{' '.join(parts[split_idx:])}"


def infer_radon_col(df: pd.DataFrame, timestamp_col: str, score_col: str, anomaly_col: str) -> str | None:
    candidates = [c for c in df.columns if c not in BASE_RESULT_COLUMNS | {timestamp_col, score_col, anomaly_col}]
    return candidates[0] if candidates else None


def load_result_series(args) -> list[ComparisonSeries]:
    labels = default_labels(args.inputs) if not args.labels else list(args.labels)
    if len(labels) != len(args.inputs):
        raise ValueError("--labels must have the same number of values as --inputs.")

    series_list = []
    for path, label in zip(args.inputs, labels):
        df = pd.read_csv(path)
        required = [args.timestamp_col, args.score_col, args.anomaly_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")

        df = df.copy()
        df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
        df = df.dropna(subset=[args.timestamp_col]).sort_values(args.timestamp_col)
        start_date, end_date = get_plot_range(args)
        if start_date is not None:
            df = df[df[args.timestamp_col] >= start_date]
        if end_date is not None:
            df = df[df[args.timestamp_col] <= end_date]
        df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
        df[args.anomaly_col] = pd.to_numeric(df[args.anomaly_col], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=[args.score_col])

        norm_score_col = f"{label}__normalized_score"
        if "threshold" in df.columns:
            df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
            valid_threshold = df["threshold"].replace(0, np.nan)
            df[norm_score_col] = df[args.score_col] / valid_threshold
            df[norm_score_col] = df[norm_score_col].replace([np.inf, -np.inf], np.nan)
            norm_score_label = "relative error (score / threshold)"
        else:
            scale = np.percentile(df[args.score_col], args.normalization_percentile)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            df[norm_score_col] = df[args.score_col] / scale
            norm_score_label = f"score / {args.normalization_percentile}th percentile"

        radon_col = infer_radon_col(df, args.timestamp_col, args.score_col, args.anomaly_col)
        series_list.append(
            ComparisonSeries(
                label=label,
                display_label=to_display_label(label),
                df=df,
                radon_col=radon_col,
                norm_score_col=norm_score_col,
                norm_score_label=norm_score_label,
            )
        )

    return series_list


def build_merged_dataframe(series_list: list[ComparisonSeries], args) -> pd.DataFrame:
    merged = None
    for item in series_list:
        keep_cols = {
            args.timestamp_col: args.timestamp_col,
            args.score_col: f"{item.label}__score",
            item.norm_score_col: f"{item.label}__normalized_score",
            args.anomaly_col: f"{item.label}__is_anomaly",
        }
        if "threshold" in item.df.columns:
            keep_cols["threshold"] = f"{item.label}__threshold"
        if "relative_error" in item.df.columns:
            keep_cols["relative_error"] = f"{item.label}__relative_error"
        if item.radon_col is not None:
            keep_cols[item.radon_col] = f"{item.label}__radon"

        current = item.df[list(keep_cols.keys())].rename(columns=keep_cols)
        if merged is None:
            merged = current
        else:
            merged = merged.merge(current, on=args.timestamp_col, how="outer")

    return merged.sort_values(args.timestamp_col)


def build_score_norm(series_list: list[ComparisonSeries], score_accessor) -> Normalize:
    score_values = []
    for item in series_list:
        values = score_accessor(item)
        if len(values) > 0:
            score_values.append(np.asarray(values, dtype=float))

    if not score_values:
        return Normalize(vmin=0.0, vmax=1.0)

    combined = np.concatenate(score_values)
    vmin = float(np.nanmin(combined))
    vmax = float(np.nanmax(combined))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def add_score_colorbar(fig, ax, norm: Normalize, label: str):
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label=label)


def build_log_intensity_norm(series_list: list[ComparisonSeries], score_accessor) -> Normalize:
    score_values = []
    for item in series_list:
        values = np.asarray(score_accessor(item), dtype=float)
        values = values[np.isfinite(values)]
        if len(values) > 0:
            score_values.append(values)

    if not score_values:
        return Normalize(vmin=0.0, vmax=1.0)

    combined = np.concatenate(score_values)
    vmax = float(np.nanmax(combined))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return Normalize(vmin=0.0, vmax=np.log1p(vmax))


def score_to_intensity(values, log_norm: Normalize) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, a_min=0.0, a_max=None)
    return np.asarray(log_norm(np.log1p(values)), dtype=float)


def lighten_color(base_color, intensity: float) -> tuple[float, float, float]:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    r, g, b = to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Low error stays very light; high error approaches the original color.
    target_lightness = 0.92 - 0.48 * intensity
    rr, gg, bb = colorsys.hls_to_rgb(h, target_lightness, s)
    return rr, gg, bb


def color_array_from_scores(base_color, values, log_norm: Normalize) -> list[tuple[float, float, float]]:
    intensities = score_to_intensity(values, log_norm)
    return [lighten_color(base_color, intensity) for intensity in intensities]


def sort_frame_for_score_drawing(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    draw_df = df.copy()
    draw_df[score_col] = pd.to_numeric(draw_df[score_col], errors="coerce")
    return draw_df.sort_values(score_col, na_position="last")


def plot_raw_scores(series_list: list[ComparisonSeries], args, output_path: str):
    with plt.rc_context(plot_rc_params(args)):
        fig, ax = plt.subplots(figsize=(args.plot_width, args.plot_height))
        log_norm = build_log_intensity_norm(series_list, lambda item: item.df[args.score_col].values)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for idx, item in enumerate(series_list):
            base_color = color_cycle[idx % len(color_cycle)] if color_cycle else "C0"
            draw_df = sort_frame_for_score_drawing(item.df, args.score_col)
            ax.plot(
                item.df[args.timestamp_col],
                item.df[args.score_col],
                color=base_color,
                linewidth=1.1,
                alpha=0.55,
            )
            ax.scatter(
                draw_df[args.timestamp_col],
                draw_df[args.score_col],
                c=color_array_from_scores(base_color, draw_df[args.score_col].values, log_norm),
                s=16,
                label=item.display_label,
            )
        ax.set_xlabel("Time")
        ax.set_ylabel(args.score_col)
        ax.set_title(f"{args.title}: raw scores")
        ax.legend()
        apply_plot_range(ax, args)
        style_time_axis(ax, args)
        apply_fixed_margins(fig)
        save_figure(fig, output_path, args)
        return fig


def plot_normalized_scores(series_list: list[ComparisonSeries], args, output_path: str):
    with plt.rc_context(plot_rc_params(args)):
        fig, ax = plt.subplots(figsize=(args.plot_width, args.plot_height))
        log_norm = build_log_intensity_norm(series_list, lambda item: item.df[item.norm_score_col].values)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for idx, item in enumerate(series_list):
            base_color = color_cycle[idx % len(color_cycle)] if color_cycle else "C0"
            draw_df = sort_frame_for_score_drawing(item.df, item.norm_score_col)
            ax.plot(
                item.df[args.timestamp_col],
                item.df[item.norm_score_col],
                color=base_color,
                linewidth=1.1,
                alpha=0.55,
            )
            ax.scatter(
                draw_df[args.timestamp_col],
                draw_df[item.norm_score_col],
                c=color_array_from_scores(base_color, draw_df[item.norm_score_col].values, log_norm),
                s=16,
                label=item.display_label,
            )
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1, label="threshold-equivalent level")
        ax.set_xlabel("Time")
        ax.set_ylabel(series_list[0].norm_score_label)
        ax.set_title(f"{args.title}: normalized scores")
        ax.legend()
        apply_plot_range(ax, args)
        style_time_axis(ax, args)
        apply_fixed_margins(fig)
        save_figure(fig, output_path, args)
        return fig


def plot_anomaly_raster(series_list: list[ComparisonSeries], args, output_path: str):
    with plt.rc_context(plot_rc_params(args)):
        height = max(2.0, float(args.raster_row_height) * len(series_list) + 0.9)
        left_margin = 0.16
        right_margin = 0.99
        fig_width = adjusted_figure_width_for_plot_box(float(args.plot_width), left_margin, right_margin, args)
        fig, ax = plt.subplots(figsize=(fig_width, height))
        log_norm = build_log_intensity_norm(
            series_list,
            lambda item: item.df.loc[item.df[args.anomaly_col] == 1, item.norm_score_col].values,
        )
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for idx, item in enumerate(series_list):
            base_color = color_cycle[idx % len(color_cycle)] if color_cycle else "C0"
            anomalies = sort_frame_for_score_drawing(
                item.df[item.df[args.anomaly_col] == 1],
                item.norm_score_col,
            )
            if not anomalies.empty:
                ax.scatter(
                    anomalies[args.timestamp_col],
                    np.full(len(anomalies), idx),
                    s=float(args.raster_dot_size),
                    c=color_array_from_scores(base_color, anomalies[item.norm_score_col].values, log_norm),
                    label=item.display_label,
                )

        ax.set_yticks(range(len(series_list)))
        ax.set_yticklabels([item.display_label for item in series_list])
        ax.set_ylim(len(series_list) - 0.5 + 0.06, -0.5 - 0.06)
        ax.tick_params(axis="y", pad=2)
        ax.margins(y=0.01)
        ax.set_xlabel("Time")
        ax.set_title(f"{args.title}: anomaly timestamps")
        ax.grid(axis="x", alpha=0.3)
        apply_plot_range(ax, args)
        style_time_axis(ax, args)
        apply_fixed_margins(fig, left=left_margin, right=right_margin, bottom=0.14, top=0.90)
        save_figure(fig, output_path, args)
        return fig


def plot_radon_overlay(series_list: list[ComparisonSeries], args, output_path: str):
    plot_candidates = [item for item in series_list if item.radon_col is not None]
    if not plot_candidates:
        return None

    with plt.rc_context(plot_rc_params(args)):
        fig = plt.figure(figsize=(args.plot_width, args.plot_height))
        for item in plot_candidates:
            plt.plot(
                item.df[args.timestamp_col],
                item.df[item.radon_col],
                label=item.display_label,
                linewidth=1.1,
            )
        plt.xlabel("Time")
        plt.ylabel("radon")
        plt.title(f"{args.title}: radon values")
        plt.legend()
        ax = plt.gca()
        apply_plot_range(ax, args)
        style_time_axis(ax, args)
        apply_fixed_margins(fig)
        save_figure(fig, output_path, args)
        return fig


def run_comparison(args):
    os.makedirs(args.output_dir, exist_ok=True)
    series_list = load_result_series(args)
    merged = build_merged_dataframe(series_list, args)

    merged_csv_path = os.path.join(args.output_dir, "comparison_merged.csv")
    raw_plot_path = os.path.join(args.output_dir, "comparison_scores_raw.png")
    normalized_plot_path = os.path.join(args.output_dir, "comparison_scores_normalized.png")
    anomaly_plot_path = os.path.join(args.output_dir, "comparison_anomalies.png")
    radon_plot_path = os.path.join(args.output_dir, "comparison_radon.png")
    pdf_path = os.path.join(args.output_dir, "comparison_plots.pdf")

    merged.to_csv(merged_csv_path, index=False)

    figs = [
        plot_raw_scores(series_list, args, raw_plot_path),
        plot_normalized_scores(series_list, args, normalized_plot_path),
        plot_anomaly_raster(series_list, args, anomaly_plot_path),
    ]

    radon_fig = plot_radon_overlay(series_list, args, radon_plot_path)
    if radon_fig is not None:
        figs.append(radon_fig)

    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)

    for fig in figs:
        plt.close(fig)

    print("\nDone.")
    print(f"Compared inputs: {len(series_list)}")
    print(f"Merged rows: {len(merged)}")
    print("\nSaved files:")
    print(f"  {merged_csv_path}")
    print(f"  {raw_plot_path}")
    print(f"  {normalized_plot_path}")
    print(f"  {anomaly_plot_path}")
    if radon_fig is not None:
        print(f"  {radon_plot_path}")
    print(f"  {pdf_path}")
