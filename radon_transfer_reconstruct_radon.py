#!/usr/bin/env python3
"""
Train radon reconstruction on selected stations and score a different target station.
"""

from __future__ import annotations

import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np
import torch

from radon_lstm_ae_torch_pkg.cli import DEFAULT_FEATURES
from radon_lstm_ae_torch_pkg.data import (
    PreparedSupervisedData,
    build_model_feature_cols,
    build_reconstruction_samples_per_file,
    chronological_train_val_split_xy,
    combine_clean_frames,
    fit_standardization_stats,
    get_device,
    load_ordered_dataframes,
    radon_model_col,
    radon_transform,
    resolve_base_feature_cols,
    set_seed,
    validate_total_rows,
)
from radon_lstm_ae_torch_pkg.reporting import print_summary, save_results_bundle
from radon_lstm_ae_torch_pkg.training import (
    compute_prediction_errors,
    make_autoencoder,
    train_supervised_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train LSTM radon reconstruction on source stations and score a target station."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="data/stations",
        help="Directory containing prepared station CSV files.",
    )
    parser.add_argument(
        "--train-stations",
        nargs="+",
        default=["blativa", "jezero", "peklo"],
        help="Station name prefixes used for training.",
    )
    parser.add_argument(
        "--test-station",
        default="loupeznicka",
        help="Station name prefix used for testing/scoring.",
    )
    parser.add_argument("--output-dir", default="transfer_reconstruct_radon_loupeznicka")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--arn-col", default="ARn(Bq/m3)")
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="Base feature columns.",
    )
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--threshold-source", default="train", choices=["train", "val"])
    parser.add_argument("--log-radon", action="store_true",
                        help="Legacy alias for --radon-transform log1p.")
    parser.add_argument("--radon-transform", default="none", choices=["none", "log1p", "sqrt"],
                        help="Transform applied to radon before modeling.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument(
        "--scaling",
        default="source",
        choices=["source", "target"],
        help=(
            "'source' standardizes the test cave with the training cave's statistics "
            "(original behavior; level offsets dominate the error). 'target' standardizes "
            "each cave with its own early-fraction statistics and calibrates the threshold "
            "on the test cave's early fraction, so the score measures whether the learned "
            "dynamics transfer."
        ),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.7,
        help="Early fraction of the test cave used for target scaling and threshold calibration.",
    )
    parser.add_argument("--add-temp-diff", action="store_true")
    parser.add_argument("--temp-outdoor-lag-hours", type=int, default=0)
    parser.add_argument("--replace-temp-outdoor-with-lag", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--plot-fontsize", type=float, default=11.0)
    parser.add_argument("--plot-width", type=float, default=14.0)
    parser.add_argument("--plot-height", type=float, default=4.2)
    parser.add_argument("--plot-dpi", type=int, default=170)
    return parser.parse_args()


def station_files(input_dir: str, stations: list[str]) -> list[str]:
    paths = []
    for station in stations:
        matches = sorted(glob.glob(os.path.join(input_dir, f"{station}*.csv")))
        if not matches:
            raise ValueError(f"No CSV files found for station '{station}' in {input_dir}.")
        paths.extend(matches)
    return paths


def load_dataframes_for_paths(args, paths: list[str], required_cols: list[str]):
    original_input = getattr(args, "input", None)
    args.input = list(paths)
    try:
        return load_ordered_dataframes(args, required_cols)
    finally:
        args.input = original_input


def prepare_transfer_data(args, train_paths: list[str], test_paths: list[str]) -> PreparedSupervisedData:
    # Resolve optional features against all files so train/test tensors have identical shapes.
    args.input = list(train_paths) + list(test_paths)
    resolve_base_feature_cols(args)
    input_feature_cols = build_model_feature_cols(args)
    target_feature_cols = [radon_model_col(args)]
    required_cols = sorted(
        set(input_feature_cols) | set(target_feature_cols) | {args.timestamp_col, args.arn_col}
    )

    train_dfs = load_dataframes_for_paths(args, train_paths, required_cols)
    test_dfs = load_dataframes_for_paths(args, test_paths, required_cols)

    _, train_input_features = combine_clean_frames(train_dfs, input_feature_cols)
    _, train_target_features = combine_clean_frames(train_dfs, target_feature_cols)
    validate_total_rows(len(train_input_features), args.window * 3, "transfer training")

    input_mean, input_std = fit_standardization_stats(
        train_input_features.values.astype(np.float32)
    )
    target_mean, target_std = fit_standardization_stats(
        train_target_features.values.astype(np.float32)
    )

    if getattr(args, "scaling", "source") == "target":
        # Standardize the test cave with its own early-fraction statistics so the
        # transfer score measures dynamics mismatch, not level offsets.
        _, test_input_features = combine_clean_frames(test_dfs, input_feature_cols)
        _, test_target_features = combine_clean_frames(test_dfs, target_feature_cols)
        n_cal = max(args.window, int(len(test_input_features) * args.calibration_fraction))
        test_input_mean, test_input_std = fit_standardization_stats(
            test_input_features.iloc[:n_cal].values.astype(np.float32)
        )
        test_target_mean, test_target_std = fit_standardization_stats(
            test_target_features.iloc[:n_cal].values.astype(np.float32)
        )
    else:
        test_input_mean, test_input_std = input_mean, input_std
        test_target_mean, test_target_std = target_mean, target_std

    _, X_train_all, y_train_all, _, _ = build_reconstruction_samples_per_file(
        dataframes=train_dfs,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
        window=args.window,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        arn_col=args.arn_col,
    )

    test_df, X_test, y_test, test_times, test_radon_values = build_reconstruction_samples_per_file(
        dataframes=test_dfs,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
        window=args.window,
        input_mean=test_input_mean,
        input_std=test_input_std,
        target_mean=test_target_mean,
        target_std=test_target_std,
        arn_col=args.arn_col,
    )

    X_train, X_val, X_train_total, y_train, y_val, y_train_total = chronological_train_val_split_xy(
        X_train_all,
        y_train_all,
        train_fraction=1.0,
        validation_size=args.validation_size,
    )

    return PreparedSupervisedData(
        df=test_df,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        X=X_test,
        y=y_test,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_train_total=X_train_total,
        y_train_total=y_train_total,
        sample_end_times=test_times,
        sample_radon_values=test_radon_values,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)

    train_paths = station_files(args.input_dir, args.train_stations)
    test_paths = station_files(args.input_dir, [args.test_station])

    prepared_data = prepare_transfer_data(args, train_paths, test_paths)
    model = make_autoencoder(
        args,
        input_size=len(prepared_data.input_feature_cols),
        output_size=len(prepared_data.target_feature_cols),
    )
    training_artifacts = train_supervised_model(
        args,
        model=model,
        prepared_data=prepared_data,
        device=device,
    )

    scores = compute_prediction_errors(
        training_artifacts.model,
        prepared_data.X,
        prepared_data.y,
        args.batch_size,
        device,
    )
    if args.scaling == "target":
        # Calibrate the anomaly threshold on the test cave's early fraction so the
        # flags mean "worse than this cave's normal fit under the transferred model".
        n_cal = max(1, int(len(scores) * args.calibration_fraction))
        train_scores = scores[:n_cal]
    else:
        train_scores = compute_prediction_errors(
            training_artifacts.model,
            prepared_data.X_train_total,
            prepared_data.y_train_total,
            args.batch_size,
            device,
        )

    model_payload = {
        "method": "transfer_reconstruct_radon",
        "model_state_dict": training_artifacts.model.state_dict(),
        "train_stations": args.train_stations,
        "test_station": args.test_station,
        "train_files": train_paths,
        "test_files": test_paths,
        "input_feature_cols": prepared_data.input_feature_cols,
        "target_feature_cols": prepared_data.target_feature_cols,
        "window": args.window,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "radon_transform": radon_transform(args),
        "scaling": args.scaling,
        "calibration_fraction": args.calibration_fraction,
    }
    scaling_payload = {
        "input_mean": prepared_data.input_mean,
        "input_std": prepared_data.input_std,
        "target_mean": prepared_data.target_mean,
        "target_std": prepared_data.target_std,
    }

    report_artifacts = save_results_bundle(
        args=args,
        history=training_artifacts.history,
        scores=scores,
        train_scores=train_scores,
        timestamps=prepared_data.sample_end_times,
        radon_values=prepared_data.sample_radon_values,
        scaling_payload=scaling_payload,
        model_payload=model_payload,
        full_df=prepared_data.df,
        method_name="Transfer radon reconstruction",
    )

    print("\nTransfer setup:")
    print(f"  Train stations: {', '.join(args.train_stations)}")
    print(f"  Test station: {args.test_station}")
    print(f"  Train files: {len(train_paths)}")
    print(f"  Test files: {len(test_paths)}")
    print_summary(args, device, prepared_data, report_artifacts)


if __name__ == "__main__":
    main()
