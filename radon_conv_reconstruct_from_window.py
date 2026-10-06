#!/usr/bin/env python3
"""
Radon anomaly detection by reconstructing radon from the full input window with Conv1D.
"""

from radon_lstm_ae_torch_pkg.cli import parse_radon_reconstruction_args
from radon_lstm_ae_torch_pkg.pipeline_conv_reconstruct_radon import run_pipeline


def main():
    args = parse_radon_reconstruction_args()
    # Use the full input window as context, but score only convolutional radon reconstruction quality.
    run_pipeline(args)


if __name__ == "__main__":
    main()
