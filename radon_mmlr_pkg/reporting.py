import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class OutputPaths:
    scores_path: str
    anomalies_path: str
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


def build_output_paths(output_dir: str) -> OutputPaths:
    return OutputPaths(
        scores_path=os.path.join(output_dir, "sequence_scores.csv"),
        anomalies_path=os.path.join(output_dir, "anomalies.csv"),
        history_path=os.path.join(output_dir, "training_history.csv"),
        model_path=os.path.join(output_dir, "model.pt"),
        scaler_path=os.path.join(output_dir, "scaling_stats.npz"),
        pdf_path=os.path.join(output_dir, "plots.pdf"),
        plot1=os.path.join(output_dir, "radon_anomalies.png"),
        plot2=os.path.join(output_dir, "reconstruction_error.png"),
        plot3=os.path.join(output_dir, "training_history.png"),
        plot4=os.path.join(output_dir, "radon_error_overlay.png"),
    )


def save_outputs_and_plots(args, artifacts) -> ReportArtifacts:
    output_paths = build_output_paths(args.output_dir)
    n_total = len(artifacts.reconstruction_error)
    n_train = max(2, int(n_total * args.train_fraction))
    threshold = float(np.percentile(artifacts.reconstruction_error[:n_train], args.threshold_percentile))

    results = pd.DataFrame(
        {
            "timestamp": artifacts.timestamps,
            "reconstruction_error": artifacts.reconstruction_error,
            "threshold": float(threshold),
            "relative_error": artifacts.reconstruction_error / max(float(threshold), 1e-12),
            "is_anomaly": artifacts.anomaly_flags.astype(int),
            args.arn_col: artifacts.radon_values,
            "prediction": artifacts.predictions,
        }
    )
    anomalies = results[results["is_anomaly"] == 1].copy()

    results.to_csv(output_paths.scores_path, index=False)
    anomalies.to_csv(output_paths.anomalies_path, index=False)
    pd.DataFrame(columns=["train_loss", "val_loss"]).to_csv(output_paths.history_path, index=False)
    with open(output_paths.model_path, "wb") as f:
        pickle.dump(artifacts.model, f)
    np.savez(output_paths.scaler_path, coefficients=artifacts.coefficients["coefficient"].values)

    save_plots(args, results, anomalies, threshold, output_paths)

    return ReportArtifacts(
        results=results,
        anomalies=anomalies,
        threshold=threshold,
        anomaly_flags=artifacts.anomaly_flags,
        output_paths=output_paths,
    )


def save_plots(args, results, anomalies, threshold: float, output_paths: OutputPaths):
    figs = []

    fig1 = plt.figure(figsize=(14, 6))
    plt.plot(results["timestamp"], results[args.arn_col], label="Radon")
    plt.plot(results["timestamp"], results["prediction"], label="MMLR fit", alpha=0.8)
    if not anomalies.empty:
        plt.scatter(anomalies["timestamp"], anomalies[args.arn_col], s=24, label="Anomalies")
    plt.xlabel("Time")
    plt.ylabel(args.arn_col)
    plt.title("MMLR radon fit with anomalies")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(output_paths.plot1, dpi=150)
    figs.append(fig1)

    fig2 = plt.figure(figsize=(14, 5))
    plt.plot(results["timestamp"], results["reconstruction_error"], label="Absolute residual")
    plt.axhline(threshold, linestyle="--", label=f"Threshold ({args.threshold_percentile}th percentile)")
    plt.xlabel("Time")
    plt.ylabel("Absolute error")
    plt.title("MMLR reconstruction error")
    plt.legend()
    plt.tight_layout()
    fig2.savefig(output_paths.plot2, dpi=150)
    figs.append(fig2)

    fig3 = plt.figure(figsize=(10, 4))
    plt.text(0.5, 0.5, "Training history is not applicable\nfor MMLR.", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    fig3.savefig(output_paths.plot3, dpi=150)
    figs.append(fig3)

    fig4, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(results["timestamp"], results[args.arn_col], label="Radon")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(args.arn_col)

    ax2 = ax1.twinx()
    ax2.plot(results["timestamp"], results["reconstruction_error"], linestyle="--", label="Absolute residual")
    ax2.axhline(threshold, linestyle=":", label="Threshold")
    ax2.set_ylabel("Error")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    plt.title("Overlay of radon values and MMLR error")
    plt.tight_layout()
    fig4.savefig(output_paths.plot4, dpi=150)
    figs.append(fig4)

    with PdfPages(output_paths.pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)

    for fig in figs:
        plt.close(fig)


def print_summary(args, device, artifacts, report_artifacts: ReportArtifacts):
    output_paths = report_artifacts.output_paths

    print("\nDone.")
    print(f"Device: {device}")
    print("Method: MMLR")
    print(f"Predictor features: {list(args.features)}")
    if getattr(args, "dropped_optional_features", []):
        print(f"Dropped optional features: {args.dropped_optional_features}")
    print(f"Added time features: {bool(args.add_time_features)}")
    print(f"Samples scored: {len(artifacts.radon_values)}")
    print(f"Anomalies detected: {int(report_artifacts.anomaly_flags.sum())}")

    print("\nSaved files:")
    print(f"  {output_paths.scores_path}")
    print(f"  {output_paths.anomalies_path}")
    print(f"  {output_paths.history_path}")
    print(f"  {output_paths.model_path}")
    print(f"  {output_paths.scaler_path}")
    print(f"  {output_paths.plot1}")
    print(f"  {output_paths.plot2}")
    print(f"  {output_paths.plot3}")
    print(f"  {output_paths.plot4}")
    print(f"  {output_paths.pdf_path}")
