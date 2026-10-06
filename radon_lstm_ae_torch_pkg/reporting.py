import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages

from radon_lstm_ae_torch_pkg.data import radon_transform
from radon_lstm_ae_torch_pkg.training import compute_prediction_errors, compute_reconstruction_errors


@dataclass
class OutputPaths:
    scores_path: str
    anomalies_path: str
    events_path: str
    history_path: str
    model_path: str
    scaler_path: str
    pdf_path: str
    plot1: str
    plot2: str
    plot3: str
    plot4: str


@dataclass
class ReportArtifacts:
    results: pd.DataFrame
    anomalies: pd.DataFrame
    threshold: float
    anomaly_flags: np.ndarray
    output_paths: OutputPaths


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


def main_plot_size(args) -> tuple[float, float]:
    return float(getattr(args, "plot_width", 14.0)), float(getattr(args, "plot_height", 4.2))


def training_plot_size(args) -> tuple[float, float]:
    width = float(getattr(args, "plot_width", 14.0))
    height = float(getattr(args, "plot_height", 4.2))
    return max(8.0, width * 0.72), max(3.6, height * 0.9)


def save_figure(fig, path: str, args):
    fig.savefig(path, dpi=int(getattr(args, "plot_dpi", 170)))


def build_output_paths(output_dir: str) -> OutputPaths:
    return OutputPaths(
        scores_path=os.path.join(output_dir, "sequence_scores.csv"),
        anomalies_path=os.path.join(output_dir, "anomalies.csv"),
        events_path=os.path.join(output_dir, "anomaly_events.csv"),
        history_path=os.path.join(output_dir, "training_history.csv"),
        model_path=os.path.join(output_dir, "model.pt"),
        scaler_path=os.path.join(output_dir, "scaling_stats.npz"),
        pdf_path=os.path.join(output_dir, "plots.pdf"),
        plot1=os.path.join(output_dir, "radon_anomalies.png"),
        plot2=os.path.join(output_dir, "reconstruction_error.png"),
        plot3=os.path.join(output_dir, "training_history.png"),
        plot4=os.path.join(output_dir, "radon_error_overlay.png"),
    )


def select_threshold_scores(args, train_scores: np.ndarray, n_fit: int) -> np.ndarray:
    """Pick which errors the anomaly threshold is fitted on.

    'train' uses every training-region error (original behavior, optimistic because the
    model was fitted on those windows). 'val' uses only the held-out validation tail,
    which better matches the error level on unseen data.
    """
    if getattr(args, "threshold_source", "train") == "val" and 0 < n_fit < len(train_scores):
        return train_scores[n_fit:]
    return train_scores


def build_anomaly_events(results: pd.DataFrame, arn_col: str) -> pd.DataFrame:
    """Merge consecutive flagged samples into events.

    Overlapping stride-1 windows flag a whole window-length run of samples per event,
    so raw flag counts overstate the number of distinct anomalies.
    """
    flags = results["is_anomaly"].values.astype(bool)
    event_id = np.cumsum(np.diff(np.concatenate([[0], flags.astype(int)])) == 1)
    flagged = results.loc[flags].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=["start", "end", "n_samples", "peak_error", "mean_error", "peak_radon", "mean_radon"]
        )
    flagged["event_id"] = event_id[flags]
    grouped = flagged.groupby("event_id")
    events = pd.DataFrame(
        {
            "start": grouped["timestamp"].first(),
            "end": grouped["timestamp"].last(),
            "n_samples": grouped.size(),
            "peak_error": grouped["reconstruction_error"].max(),
            "mean_error": grouped["reconstruction_error"].mean(),
            "peak_radon": grouped[arn_col].max(),
            "mean_radon": grouped[arn_col].mean(),
        }
    )
    return events.reset_index(drop=True)


def save_results_bundle(
    args,
    history: dict[str, list[float]],
    scores: np.ndarray,
    train_scores: np.ndarray,
    timestamps: pd.Index,
    radon_values: np.ndarray,
    scaling_payload: dict,
    model_payload: dict,
    full_df: pd.DataFrame,
    method_name: str,
    n_fit: int = 0,
) -> ReportArtifacts:
    threshold_scores = select_threshold_scores(args, train_scores, n_fit)
    threshold = np.percentile(threshold_scores, args.threshold_percentile)
    anomaly_flags = scores > threshold

    results = pd.DataFrame(
        {
            "timestamp": timestamps,
            "reconstruction_error": scores,
            "threshold": float(threshold),
            "relative_error": scores / max(float(threshold), 1e-12),
            "is_anomaly": anomaly_flags.astype(int),
            args.arn_col: radon_values,
        }
    )
    anomalies = results[results["is_anomaly"] == 1].copy()
    events = build_anomaly_events(results, args.arn_col)

    output_paths = build_output_paths(args.output_dir)
    results.to_csv(output_paths.scores_path, index=False)
    anomalies.to_csv(output_paths.anomalies_path, index=False)
    events.to_csv(output_paths.events_path, index=False)
    pd.DataFrame(history).to_csv(output_paths.history_path, index=False)
    torch.save(model_payload, output_paths.model_path)
    np.savez(output_paths.scaler_path, **scaling_payload)

    save_plots(
        args=args,
        df=full_df,
        history=history,
        results=results,
        anomalies=anomalies,
        threshold=threshold,
        output_paths=output_paths,
        score_label=f"{method_name} error",
    )

    return ReportArtifacts(
        results=results,
        anomalies=anomalies,
        threshold=threshold,
        anomaly_flags=anomaly_flags,
        output_paths=output_paths,
    )

def save_outputs_and_plots(
    args,
    prepared_data,
    training_artifacts,
    device: torch.device,
    method_key: str = "reconstruct_all_features",
    method_name: str = "Reconstruction",
    extra_model_payload: dict | None = None,
) -> ReportArtifacts:
    model = training_artifacts.model
    history = training_artifacts.history

    mse_all = compute_reconstruction_errors(model, prepared_data.X, args.batch_size, device)
    mse_train = compute_reconstruction_errors(model, prepared_data.X_train_total, args.batch_size, device)

    model_payload = {
        "method": method_key,
        "model_state_dict": model.state_dict(),
        "feature_cols": prepared_data.feature_cols,
        "window": args.window,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "radon_transform": radon_transform(args),
    }
    if hasattr(args, "conv_kernel_size"):
        model_payload["conv_kernel_size"] = args.conv_kernel_size
        model_payload["conv_temporal_bottleneck"] = getattr(args, "conv_temporal_bottleneck", False)
    if hasattr(args, "transformer_nhead"):
        model_payload["transformer_nhead"] = args.transformer_nhead
        model_payload["transformer_dim_feedforward"] = args.transformer_dim_feedforward
    if extra_model_payload:
        model_payload.update(extra_model_payload)
    scaling_payload = {"mean": prepared_data.mean, "std": prepared_data.std}
    radon_values = prepared_data.df.loc[prepared_data.seq_end_times, args.arn_col].values

    return save_results_bundle(
        args=args,
        history=history,
        scores=mse_all,
        train_scores=mse_train,
        timestamps=prepared_data.seq_end_times,
        radon_values=radon_values,
        scaling_payload=scaling_payload,
        model_payload=model_payload,
        full_df=prepared_data.df,
        method_name=method_name,
        n_fit=len(prepared_data.X_train),
    )


def save_supervised_outputs_and_plots(
    args,
    prepared_data,
    training_artifacts,
    device: torch.device,
    method_key: str,
    method_name: str,
    extra_model_payload: dict | None = None,
) -> ReportArtifacts:
    model = training_artifacts.model
    history = training_artifacts.history

    scores = compute_prediction_errors(
        model,
        prepared_data.X,
        prepared_data.y,
        args.batch_size,
        device,
    )
    train_scores = compute_prediction_errors(
        model,
        prepared_data.X_train_total,
        prepared_data.y_train_total,
        args.batch_size,
        device,
    )

    model_payload = {
        "method": method_key,
        "model_state_dict": model.state_dict(),
        "input_feature_cols": prepared_data.input_feature_cols,
        "target_feature_cols": prepared_data.target_feature_cols,
        "window": args.window,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "radon_transform": radon_transform(args),
    }
    if hasattr(args, "conv_kernel_size"):
        model_payload["conv_kernel_size"] = args.conv_kernel_size
        model_payload["conv_temporal_bottleneck"] = getattr(args, "conv_temporal_bottleneck", False)
    if hasattr(args, "transformer_nhead"):
        model_payload["transformer_nhead"] = args.transformer_nhead
        model_payload["transformer_dim_feedforward"] = args.transformer_dim_feedforward
    if hasattr(args, "forecast_horizon"):
        model_payload["forecast_horizon"] = args.forecast_horizon
    if extra_model_payload:
        model_payload.update(extra_model_payload)

    scaling_payload = {
        "input_mean": prepared_data.input_mean,
        "input_std": prepared_data.input_std,
        "target_mean": prepared_data.target_mean,
        "target_std": prepared_data.target_std,
    }

    return save_results_bundle(
        args=args,
        history=history,
        scores=scores,
        train_scores=train_scores,
        timestamps=prepared_data.sample_end_times,
        radon_values=prepared_data.sample_radon_values,
        scaling_payload=scaling_payload,
        model_payload=model_payload,
        full_df=prepared_data.df,
        method_name=method_name,
        n_fit=len(prepared_data.X_train),
    )


def save_plots(args, df, history, results, anomalies, threshold, output_paths: OutputPaths, score_label: str):
    figs = []
    marker_size = max(26.0, float(getattr(args, "plot_fontsize", 11.0)) * 3.0)

    with plt.rc_context(plot_rc_params(args)):
        fig1 = plt.figure(figsize=main_plot_size(args))
        plt.plot(df.index, df[args.arn_col], label=args.arn_col, linewidth=1.3)
        if not anomalies.empty:
            plt.scatter(
                anomalies["timestamp"],
                anomalies[args.arn_col],
                s=marker_size,
                label="Anomalies",
                zorder=3,
            )
        plt.xlabel("Time")
        plt.ylabel(args.arn_col)
        plt.title("Radon with detected anomalies")
        plt.legend()
        plt.tight_layout()
        save_figure(fig1, output_paths.plot1, args)
        figs.append(fig1)

        fig2 = plt.figure(figsize=main_plot_size(args))
        plt.plot(results["timestamp"], results["reconstruction_error"], label=score_label, linewidth=1.3)
        plt.axhline(threshold, linestyle="--", label=f"Threshold ({args.threshold_percentile}th percentile)")
        plt.xlabel("Time")
        plt.ylabel("MSE")
        plt.title(f"{method_name_to_title(score_label)} over time")
        plt.legend()
        plt.tight_layout()
        save_figure(fig2, output_paths.plot2, args)
        figs.append(fig2)

        fig3 = plt.figure(figsize=training_plot_size(args))
        plt.plot(history["train_loss"], label="Train loss", linewidth=1.3)
        if len(history["val_loss"]) > 0:
            plt.plot(history["val_loss"], label="Validation loss", linewidth=1.3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training history")
        plt.legend()
        plt.tight_layout()
        save_figure(fig3, output_paths.plot3, args)
        figs.append(fig3)

        fig4, ax1 = plt.subplots(figsize=main_plot_size(args))
        ax1.plot(results["timestamp"], results[args.arn_col], label="Radon", linewidth=1.3)
        ax1.set_xlabel("Time")
        ax1.set_ylabel(args.arn_col)

        ax2 = ax1.twinx()
        ax2.plot(results["timestamp"], results["reconstruction_error"], linestyle="--", label=score_label, linewidth=1.2)
        ax2.axhline(threshold, linestyle=":", label="Threshold")
        ax2.set_ylabel("Error")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

        plt.title("Overlay of radon values and anomaly score")
        plt.tight_layout()
        save_figure(fig4, output_paths.plot4, args)
        figs.append(fig4)

    with PdfPages(output_paths.pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)

    for fig in figs:
        plt.close(fig)


def method_name_to_title(score_label: str) -> str:
    return score_label[:1].upper() + score_label[1:]




def print_summary(args, device, prepared_data, report_artifacts: ReportArtifacts):
    output_paths = report_artifacts.output_paths

    print("\nDone.")
    print(f"Device: {device}")
    if hasattr(prepared_data, "feature_cols"):
        print(f"Used features: {prepared_data.feature_cols}")
    else:
        print(f"Input features: {prepared_data.input_feature_cols}")
        print(f"Target features: {prepared_data.target_feature_cols}")
    if getattr(args, "dropped_optional_features", []):
        print(f"Dropped optional features: {args.dropped_optional_features}")
    print(f"Window size: {args.window}")
    if hasattr(args, "forecast_horizon"):
        print(f"Forecast horizon: {args.forecast_horizon}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Latent size: {args.latent_size}")
    train_count = len(prepared_data.X_train_total)
    all_count = len(prepared_data.X)
    print(f"Training samples: {train_count}")
    print(f"All samples scored: {all_count}")
    print(f"Threshold: {report_artifacts.threshold:.6f}")
    print(f"Threshold source: {getattr(args, 'threshold_source', 'train')}")
    flags = report_artifacts.anomaly_flags.astype(int)
    n_events = int((np.diff(np.concatenate([[0], flags])) == 1).sum())
    print(f"Anomalous samples: {int(flags.sum())} (merged into {n_events} contiguous events)")

    print("\nSaved files:")
    print(f"  {output_paths.scores_path}")
    print(f"  {output_paths.anomalies_path}")
    print(f"  {output_paths.events_path}")
    print(f"  {output_paths.history_path}")
    print(f"  {output_paths.model_path}")
    print(f"  {output_paths.scaler_path}")
    print(f"  {output_paths.plot1}")
    print(f"  {output_paths.plot2}")
    print(f"  {output_paths.plot3}")
    print(f"  {output_paths.plot4}")
    print(f"  {output_paths.pdf_path}")
