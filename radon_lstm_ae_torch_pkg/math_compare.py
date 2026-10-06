from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from radon_lstm_ae_torch_pkg.compare import ComparisonSeries, default_labels, load_result_series


@dataclass
class Event:
    start: pd.Timestamp
    end: pd.Timestamp
    count: int


def parse_math_compare_args():
    parser = argparse.ArgumentParser(
        description="Compute mathematical agreement metrics between anomaly-detection result CSVs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="Result CSV files, typically sequence_scores.csv outputs.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels for the inputs. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--station-dir",
        default=None,
        help=(
            "Optional station output directory like parameter_suite_runs/by_station/loupeznicka. "
            "If set, sequence_scores.csv files are discovered automatically from method subdirectories."
        ),
    )
    parser.add_argument(
        "--suite-root",
        default="parameter_suite_runs/by_station",
        help=(
            "Root directory that contains per-station subdirectories. Used with "
            "--method-name to compare one method across stations."
        ),
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help=(
            "Optional method name like reconstruct_radon or mmlr. If set, "
            "sequence_scores.csv files are discovered across station subdirectories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_math_results",
        help="Directory for pairwise metric CSVs.",
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
        help="Percentile used to scale each score series when threshold is absent.",
    )
    parser.add_argument(
        "--bridge-gap-hours",
        type=float,
        default=2.0,
        help="Merge anomalous timestamps into one event when the gap is at most this many hours.",
    )
    parser.add_argument(
        "--event-tolerance-hours",
        type=float,
        default=6.0,
        help="Tolerance for matching events across methods.",
    )
    parser.add_argument(
        "--daily-align",
        action="store_true",
        help=(
            "Align score series by calendar day instead of exact timestamps. "
            "Useful for cross-station comparisons when stations do not share identical time stamps."
        ),
    )
    return parser.parse_args()


def resolve_inputs(args):
    if args.station_dir:
        inputs = []
        labels = []
        for entry in sorted(os.scandir(args.station_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            csv_path = os.path.join(entry.path, "sequence_scores.csv")
            if os.path.isfile(csv_path):
                inputs.append(csv_path)
                labels.append(entry.name)
        if len(inputs) < 2:
            raise ValueError(f"Need at least 2 method outputs in {args.station_dir}.")
        args.inputs = inputs
        args.labels = labels
    elif args.method_name:
        inputs = []
        labels = []
        for entry in sorted(os.scandir(args.suite_root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            csv_path = os.path.join(entry.path, args.method_name, "sequence_scores.csv")
            if os.path.isfile(csv_path):
                inputs.append(csv_path)
                labels.append(entry.name)
        if len(inputs) < 2:
            raise ValueError(
                f"Need at least 2 station outputs for method '{args.method_name}' in {args.suite_root}."
            )
        args.inputs = inputs
        args.labels = labels
    elif not args.inputs:
        raise ValueError("Provide one of: --inputs, --station-dir, or --method-name.")


def aligned_pair_frame(a: ComparisonSeries, b: ComparisonSeries, args) -> pd.DataFrame:
    if args.method_name or args.daily_align:
        return aligned_pair_frame_daily(a, b, args)

    left = a.df[[args.timestamp_col, a.norm_score_col, args.anomaly_col]].rename(
        columns={
            a.norm_score_col: "score_a",
            args.anomaly_col: "anomaly_a",
        }
    )
    right = b.df[[args.timestamp_col, b.norm_score_col, args.anomaly_col]].rename(
        columns={
            b.norm_score_col: "score_b",
            args.anomaly_col: "anomaly_b",
        }
    )
    merged = left.merge(right, on=args.timestamp_col, how="inner").sort_values(args.timestamp_col)
    merged["score_a"] = pd.to_numeric(merged["score_a"], errors="coerce")
    merged["score_b"] = pd.to_numeric(merged["score_b"], errors="coerce")
    merged["anomaly_a"] = pd.to_numeric(merged["anomaly_a"], errors="coerce").fillna(0).astype(int)
    merged["anomaly_b"] = pd.to_numeric(merged["anomaly_b"], errors="coerce").fillna(0).astype(int)
    return merged.dropna(subset=["score_a", "score_b"])


def aligned_pair_frame_daily(a: ComparisonSeries, b: ComparisonSeries, args) -> pd.DataFrame:
    left = a.df[[args.timestamp_col, a.norm_score_col, args.anomaly_col]].copy()
    right = b.df[[args.timestamp_col, b.norm_score_col, args.anomaly_col]].copy()

    left["day"] = pd.to_datetime(left[args.timestamp_col], utc=True, errors="coerce").dt.floor("D")
    right["day"] = pd.to_datetime(right[args.timestamp_col], utc=True, errors="coerce").dt.floor("D")

    left = (
        left.groupby("day", as_index=False)
        .agg(
            score_a=(a.norm_score_col, "mean"),
            anomaly_a=(args.anomaly_col, "max"),
        )
        .dropna(subset=["score_a"])
    )
    right = (
        right.groupby("day", as_index=False)
        .agg(
            score_b=(b.norm_score_col, "mean"),
            anomaly_b=(args.anomaly_col, "max"),
        )
        .dropna(subset=["score_b"])
    )

    merged = left.merge(right, on="day", how="inner").sort_values("day")
    merged["score_a"] = pd.to_numeric(merged["score_a"], errors="coerce")
    merged["score_b"] = pd.to_numeric(merged["score_b"], errors="coerce")
    merged["anomaly_a"] = pd.to_numeric(merged["anomaly_a"], errors="coerce").fillna(0).astype(int)
    merged["anomaly_b"] = pd.to_numeric(merged["anomaly_b"], errors="coerce").fillna(0).astype(int)
    return merged.dropna(subset=["score_a", "score_b"])


def jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a == 1, b == 1).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a == 1, b == 1).sum()
    return float(inter / union)


def build_events(df: pd.DataFrame, timestamp_col: str, anomaly_col: str, bridge_gap_hours: float) -> list[Event]:
    anomalies = df[df[anomaly_col] == 1].sort_values(timestamp_col)
    if anomalies.empty:
        return []

    gap = pd.Timedelta(hours=bridge_gap_hours)
    times = list(anomalies[timestamp_col])
    events = []
    start = times[0]
    prev = times[0]
    count = 1

    for ts in times[1:]:
        if ts - prev <= gap:
            prev = ts
            count += 1
            continue
        events.append(Event(start=start, end=prev, count=count))
        start = ts
        prev = ts
        count = 1
    events.append(Event(start=start, end=prev, count=count))
    return events


def events_match(a: Event, b: Event, tolerance: pd.Timedelta) -> bool:
    zero = pd.Timedelta(0)
    if a.start <= b.end and b.start <= a.end:
        return True
    if abs(a.start - b.start) <= tolerance:
        return True
    if abs(a.end - b.end) <= tolerance:
        return True
    if zero <= (b.start - a.end) <= tolerance:
        return True
    if zero <= (a.start - b.end) <= tolerance:
        return True
    return False


def greedy_match_events(events_a: list[Event], events_b: list[Event], tolerance_hours: float):
    tolerance = pd.Timedelta(hours=tolerance_hours)
    candidates = []
    for idx_a, event_a in enumerate(events_a):
        for idx_b, event_b in enumerate(events_b):
            if events_match(event_a, event_b, tolerance):
                start_diff = abs((event_a.start - event_b.start).total_seconds()) / 3600.0
                candidates.append((start_diff, idx_a, idx_b))
    candidates.sort()

    used_a = set()
    used_b = set()
    matches = []
    for start_diff, idx_a, idx_b in candidates:
        if idx_a in used_a or idx_b in used_b:
            continue
        used_a.add(idx_a)
        used_b.add(idx_b)
        matches.append((idx_a, idx_b, start_diff))
    return matches


def event_metrics(events_a: list[Event], events_b: list[Event], tolerance_hours: float) -> dict[str, float]:
    matches = greedy_match_events(events_a, events_b, tolerance_hours)
    tp = len(matches)
    n_a = len(events_a)
    n_b = len(events_b)
    precision = float(tp / n_a) if n_a else 1.0
    recall = float(tp / n_b) if n_b else 1.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    start_diffs = [match[2] for match in matches]
    median_start_diff = float(np.median(start_diffs)) if start_diffs else np.nan
    return {
        "event_matches": tp,
        "events_a": n_a,
        "events_b": n_b,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "median_start_diff_hours": median_start_diff,
    }


def compare_pair(a: ComparisonSeries, b: ComparisonSeries, args) -> dict[str, float | str]:
    aligned = aligned_pair_frame(a, b, args)
    spearman = aligned["score_a"].corr(aligned["score_b"], method="spearman")
    pearson = aligned["score_a"].corr(aligned["score_b"], method="pearson")
    jaccard = jaccard_binary(aligned["anomaly_a"].to_numpy(), aligned["anomaly_b"].to_numpy())

    events_a = build_events(a.df, args.timestamp_col, args.anomaly_col, args.bridge_gap_hours)
    events_b = build_events(b.df, args.timestamp_col, args.anomaly_col, args.bridge_gap_hours)
    event_summary = event_metrics(events_a, events_b, args.event_tolerance_hours)

    return {
        "item_a": a.label,
        "item_b": b.label,
        "aligned_rows": len(aligned),
        "spearman_relative_error": float(spearman) if pd.notna(spearman) else np.nan,
        "pearson_relative_error": float(pearson) if pd.notna(pearson) else np.nan,
        "jaccard_anomaly_hours": jaccard,
        "bridge_gap_hours": float(args.bridge_gap_hours),
        "event_tolerance_hours": float(args.event_tolerance_hours),
        **event_summary,
    }


def run_math_comparison(args):
    resolve_inputs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    series_list = load_result_series(args)

    rows = []
    for i, a in enumerate(series_list):
        for b in series_list[i + 1 :]:
            rows.append(compare_pair(a, b, args))

    results = pd.DataFrame(rows).sort_values(["item_a", "item_b"]).reset_index(drop=True)
    out_csv = os.path.join(args.output_dir, "pairwise_method_metrics.csv")
    results.to_csv(out_csv, index=False)

    if args.station_dir:
        station_name = os.path.basename(os.path.normpath(args.station_dir))
        matrix = results.pivot(index="item_a", columns="item_b", values="event_f1")
        matrix.to_csv(os.path.join(args.output_dir, f"{station_name}_event_f1_matrix.csv"))
    elif args.method_name:
        matrix = results.pivot(index="item_a", columns="item_b", values="event_f1")
        matrix.to_csv(os.path.join(args.output_dir, f"{args.method_name}_event_f1_matrix.csv"))

    print("\nDone.")
    if args.method_name:
        print(f"Compared stations: {len(series_list)}")
        print(f"Method: {args.method_name}")
    else:
        print(f"Compared methods: {len(series_list)}")
    print(f"Pairwise rows: {len(results)}")
    print(f"Saved files:\n  {out_csv}")
