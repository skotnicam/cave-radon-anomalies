#!/usr/bin/env bash
# Run every experiment of the paper in order, then the analyses and figures.
# Usage: experiments/run_all.sh [runs_root]      (default runs/; RESUME=1 skips finished model runs)
# Approximate CPU time on a laptop: 01 ~75 min, 02 ~20 min, 04 ~60 min, the rest ~60 min together.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
for s in 01_main_suite 02_radim_suite 03_ochtinska 04_station_transfer 05_multistation 06_cave_transfer \
         07_probe_comparison 08_radim_leaveout 09_precip 10_meteo_latent; do
  experiments/$s.sh "$ROOT"
done
./make_figures.sh "$ROOT"
