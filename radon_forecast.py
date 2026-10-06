#!/usr/bin/env python3
"""
Radon anomaly detection by forecasting future radon steps from the input window.
"""

from radon_lstm_ae_torch_pkg.cli import parse_forecast_args
from radon_lstm_ae_torch_pkg.pipeline_forecast import run_pipeline


def main():
    args = parse_forecast_args()
    # Forecast future radon values and use forecast error as the anomaly score.
    run_pipeline(args)


if __name__ == "__main__":
    main()
