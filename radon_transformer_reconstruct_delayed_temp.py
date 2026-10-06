#!/usr/bin/env python3
"""
Radon anomaly detection with a Transformer and delayed outdoor temperature.
"""

from radon_lstm_ae_torch_pkg.cli import parse_transformer_delayed_temp_reconstruction_args
from radon_lstm_ae_torch_pkg.pipeline_transformer_reconstruct_radon_delayed_temp import run_pipeline


def main():
    args = parse_transformer_delayed_temp_reconstruction_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
