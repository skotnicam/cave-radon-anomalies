#!/usr/bin/env bash
# Regenerate every figure and table of the paper from a runs root (default runs/) into <runs_root>/figures/
# and copy the PDFs to figures/. Run from the repository root after experiments/run_all.sh (or a subset:
# each analysis only needs the experiments listed next to it).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="${1:-runs}"
mkdir -p "$ROOT/figures"
run() { echo "== $1"; python -W ignore "analysis/$1" "${@:2}" || echo "FAILED: $1"; }
run make_figures.py --root "$ROOT"       # 01 (+02): method_summary, method_agreement, station_consensus, negative_examples
run method_agreement.py "$ROOT"          # 01: median Jaccard between methods (text)
run training_curves.py "$ROOT"           # 01: training_curves (Appendix B)
run station_structure.py "$ROOT"         # data only: station_structure
run pressure_lows.py "$ROOT"             # 01, 02: pressure_epoch (Appendix C)
run rain_error.py "$ROOT"                # 01: boris_error, rain_epoch
run precip_summary.py "$ROOT"            # 01, 09: Boris with precipitation as input (text)
run meteo_latent_summary.py "$ROOT"      # 01, 10: larger latent space (text)
run station_transfer.py "$ROOT"          # 01, 04: station_transfer
run multistation_table.py "$ROOT"        # 01, 04, 05: Table 4
run cave_transfer.py "$ROOT"             # 02, 03, 06: cave transfer tables
run ochtinska_figure.py "$ROOT"          # 02, 03, 06: ochtinska (Appendix D)
run probe_comparison.py "$ROOT"          # 01, 02, 07: pairs.csv
run two_probes_figure.py "$ROOT"         # 01, 07: two_probes (Appendix F)
run microseisms.py "$ROOT"               # 01, 02: microseisms (Appendix E)
run seismic_catalogues.py "$ROOT"        # 01, 03: catalogue tests (text)
run seismic_radim.py "$ROOT"             # 02: qualifying earthquakes, RADIM tests (text)
run radim_leaveout.py "$ROOT"            # 02, 08: leave-event-out control (text)
mkdir -p figures && cp "$ROOT"/figures/*.pdf figures/ 2>/dev/null
echo "figures copied to figures/ from $ROOT/figures/"
