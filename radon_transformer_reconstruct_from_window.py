#!/usr/bin/env python3
"""
Radon anomaly detection by reconstructing radon with a Transformer sequence autoencoder.
"""

from radon_lstm_ae_torch_pkg.cli import parse_transformer_radon_reconstruction_args
from radon_lstm_ae_torch_pkg.pipeline_transformer_reconstruct_radon import run_pipeline


def main():
    args = parse_transformer_radon_reconstruction_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
