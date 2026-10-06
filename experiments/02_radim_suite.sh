#!/usr/bin/env bash
# The ten methods on the nine-year record of the RADIM reference monitor at the Jezero site (2017-2026).
# Same settings as the main suite; the training region (first 70 %) ends in August 2023.
#
# Usage:  experiments/02_radim_suite.sh [runs_root]
# Output: <runs_root>/radim_suite/by_station/bozkov2017/<method>/
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
STATIONS=bozkov2017 INPUT_DIR=data/radim_jezero SUITE_OUT="$ROOT/radim_suite" experiments/01_main_suite.sh "$ROOT"
