import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect anomalies with an EMD + SVR baseline."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more input CSV files.",
    )
    parser.add_argument("--output-dir", default="results_emd_svr", help="Directory for outputs.")
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--arn-col", default="ARn(Bq/m3)", help="Radon column name.")
    parser.add_argument("--svr-c", type=float, default=1.0, help="SVR C parameter.")
    parser.add_argument("--svr-epsilon", type=float, default=0.1, help="SVR epsilon parameter.")
    parser.add_argument(
        "--svr-gamma",
        default="scale",
        help="SVR gamma parameter, for example 'scale', 'auto', or a float value.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Unused for EMD-SVR, kept for output consistency.",
    )
    return parser.parse_args()
