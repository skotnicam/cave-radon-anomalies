#!/usr/bin/env python3
"""
Radon anomaly detection by forecasting future radon steps with Conv1D.
"""

from radon_lstm_ae_torch_pkg.cli import parse_forecast_args
from radon_lstm_ae_torch_pkg.pipeline_conv_forecast import run_pipeline


def main():
    args = parse_forecast_args()
    # Forecast future radon values with a convolutional encoder and score forecast error.
    run_pipeline(args)


if __name__ == "__main__":
    main()
