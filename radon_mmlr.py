#!/usr/bin/env python3
"""
Radon anomaly detection with a multiple linear regression baseline.
"""

from radon_mmlr_pkg.cli import parse_args
from radon_mmlr_pkg.pipeline import run_pipeline


def main():
    args = parse_args()
    # Linear baseline inspired by the MATLAB fitlm(..., 'interactions') workflow.
    run_pipeline(args)


if __name__ == "__main__":
    main()
