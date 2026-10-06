#!/usr/bin/env python3
"""
Radon anomaly detection with an EMD + SVR baseline.
"""

from radon_emd_svr_pkg.cli import parse_args
from radon_emd_svr_pkg.pipeline import run_pipeline


def main():
    args = parse_args()
    # Port of the MATLAB EMD-SVR idea with outputs aligned to the rest of this repository.
    run_pipeline(args)


if __name__ == "__main__":
    main()
