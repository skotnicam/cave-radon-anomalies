#!/usr/bin/env python3
"""
Radon anomaly detection by reconstructing radon from meteorological inputs.
"""

from radon_lstm_ae_torch_pkg.cli import parse_meteo_to_radon_args
from radon_lstm_ae_torch_pkg.pipeline_meteo_to_radon import run_pipeline


def main():
    args = parse_meteo_to_radon_args()
    # Exclude radon from the inputs so anomalies reflect poor meteo-to-radon explainability.
    run_pipeline(args)


if __name__ == "__main__":
    main()
