import argparse


DEFAULT_FEATURES = [
    "temperature_indoor",
    "temperature_outdoor",
    "humidity_indoor",
    "pressure",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect anomalies with a multiple linear regression baseline."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input CSV files.",
    )
    parser.add_argument("--output-dir", default="results_mmlr", help="Directory for outputs.")
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--arn-col", default="ARn(Bq/m3)", help="Radon column name.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_FEATURES),
        help="Predictor columns used for the regression baseline. Pressure is used by default in place of humidity_outdoor. Optional columns such as humidity_outdoor are dropped automatically if absent.",
    )
    parser.add_argument(
        "--add-time-features",
        action="store_true",
        help="Add cyclical hour-of-day and day-of-year features to the regression baseline.",
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
            "Transform applied to radon before fitting; predictions are reported in raw "
            "units. Prevents negative predictions and reduces the correlation between "
            "residual size and radon level ('sqrt' stabilizes counting noise)."
        ),
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold on absolute residuals.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Fraction of earliest samples used to fit the regression model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Unused for MMLR, kept for output consistency.",
    )
    return parser.parse_args()
