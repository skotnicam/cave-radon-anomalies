import argparse


DEFAULT_FEATURES = [
    "ARn(Bq/m3)",
    "temperature_indoor",
    "temperature_outdoor",
    "pressure",
]


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input CSV files. Windows are created within each file only.",
    )
    parser.add_argument("--output-dir", default="results", help="Directory for outputs.")
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--arn-col", default="ARn(Bq/m3)", help="Radon column name.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="Base feature columns. Pressure is used by default in place of humidity_outdoor. Optional columns such as humidity_outdoor are dropped automatically if absent.",
    )
    parser.add_argument("--window", type=int, default=12, help="Sequence length.")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--latent-size", type=int, default=32, help="Latent size.")
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=5,
        help="Kernel size used by the convolutional variants.",
    )
    parser.add_argument(
        "--conv-temporal-bottleneck",
        action="store_true",
        help=(
            "Use stride-2 convolutions in the conv autoencoder so time is compressed 4x. "
            "Without this the conv AE can pass the input through per-timestep channels and "
            "reconstruct anomalies perfectly (near-zero error in suite runs)."
        ),
    )
    parser.add_argument(
        "--transformer-nhead",
        type=int,
        default=4,
        help="Number of attention heads used by the Transformer variants.",
    )
    parser.add_argument(
        "--transformer-dim-feedforward",
        type=int,
        default=128,
        help="Feed-forward dimension used inside Transformer encoder layers.",
    )
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="LSTM dropout.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 weight decay passed to Adam. Helps against the train/val gap seen in suite runs.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Clip gradient norm to this value during training (0 disables clipping).",
    )
    parser.add_argument(
        "--log-radon",
        action="store_true",
        help="Legacy alias for --radon-transform log1p.",
    )
    parser.add_argument(
        "--radon-transform",
        default="none",
        choices=["none", "log1p", "sqrt"],
        help=(
            "Transform applied to radon before modeling. 'log1p' makes errors relative but "
            "over-weights noisy low-radon periods; 'sqrt' stabilizes counting noise and sits "
            "between raw (error tracks amplitude) and log."
        ),
    )
    parser.add_argument("--validation-size", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Minimum number of epochs to run before early stopping can trigger.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of non-improving validation epochs allowed before early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of earliest samples for training.",
    )
    parser.add_argument(
        "--train-region",
        choices=["head", "tail"],
        default="head",
        help="Which end of the record is the training region: head = earliest fraction (default), tail = latest fraction.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold on training error.",
    )
    parser.add_argument(
        "--threshold-source",
        default="train",
        choices=["train", "val"],
        help=(
            "Errors used to fit the anomaly threshold. 'train' reproduces the original "
            "behavior; 'val' uses held-out validation errors, which avoids the optimistic "
            "in-sample threshold that inflates anomaly rates on later data."
        ),
    )
    parser.add_argument("--add-temp-diff", action="store_true", help="Add temp_diff feature.")
    parser.add_argument(
        "--temp-outdoor-lag-hours",
        type=int,
        default=0,
        help="Use an outdoor-temperature feature delayed by this many hourly steps.",
    )
    parser.add_argument(
        "--replace-temp-outdoor-with-lag",
        action="store_true",
        help="Replace temperature_outdoor with its delayed version instead of appending a new lagged feature.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device selection.",
    )
    parser.add_argument(
        "--plot-fontsize",
        type=float,
        default=11.0,
        help="Base font size used in output plots.",
    )
    parser.add_argument(
        "--plot-width",
        type=float,
        default=14.0,
        help="Plot width in inches for the main output figures.",
    )
    parser.add_argument(
        "--plot-height",
        type=float,
        default=4.2,
        help="Plot height in inches for the main output figures.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=170,
        help="DPI used when saving output plots.",
    )
    return parser


def parse_args():
    parser = build_common_parser(
        "Detect anomalies in cave radon data using a PyTorch LSTM autoencoder."
    )
    return parser.parse_args()


def parse_radon_reconstruction_args():
    parser = build_common_parser(
        "Detect anomalies by reconstructing radon from the full input window."
    )
    return parser.parse_args()


def parse_meteo_to_radon_args():
    parser = build_common_parser(
        "Detect anomalies by reconstructing radon from meteorological inputs only."
    )
    parser.set_defaults(min_epochs=10, early_stopping_patience=12)
    return parser.parse_args()


def parse_forecast_args():
    parser = build_common_parser(
        "Detect anomalies by forecasting future radon values from the input window."
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=3,
        help="Number of future steps to predict for each input window.",
    )
    return parser.parse_args()


def parse_delayed_temp_reconstruction_args():
    parser = build_common_parser(
        "Detect anomalies by reconstructing radon from the full input window with delayed outdoor temperature."
    )
    parser.set_defaults(temp_outdoor_lag_hours=6, replace_temp_outdoor_with_lag=True)
    return parser.parse_args()


def parse_transformer_radon_reconstruction_args():
    parser = build_common_parser(
        "Detect anomalies by reconstructing radon with a Transformer sequence autoencoder."
    )
    return parser.parse_args()


def parse_transformer_delayed_temp_reconstruction_args():
    parser = build_common_parser(
        "Detect anomalies with a Transformer radon reconstruction model using delayed outdoor temperature."
    )
    parser.set_defaults(temp_outdoor_lag_hours=6, replace_temp_outdoor_with_lag=True)
    return parser.parse_args()
