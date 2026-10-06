#!/usr/bin/env python3
"""
Radon anomaly detection by reconstructing radon from a full window with delayed outdoor temperature.
"""

from radon_lstm_ae_torch_pkg.cli import parse_delayed_temp_reconstruction_args
from radon_lstm_ae_torch_pkg.pipeline_reconstruct_radon_delayed_temp import run_pipeline


def main():
    args = parse_delayed_temp_reconstruction_args()
    # Use a lagged outdoor-temperature feature to test delayed environmental influence on radon.
    run_pipeline(args)


if __name__ == "__main__":
    main()
