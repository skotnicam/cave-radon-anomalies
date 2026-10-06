#!/usr/bin/env bash
# Two instruments at the Jezero site (Section 4.5, Appendix F): LSTM reconstruction retrained with other seeds,
# on the common period of the two probes (March 2024 - March 2026, training to 21 August 2025), and on the long
# RADIM record. The canonical seed-42 runs come from the main and the RADIM suite.
#
# Usage:  experiments/07_probe_comparison.sh [runs_root]      (needs 01 and 02 finished)
# Output: <runs_root>/probe_comparison/<run>/ and pairs.csv (analysis/probe_comparison.py)
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/probe_comparison"; mkdir -p "$OUT/logs"
COMMON=(--window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 1e-3
        --min-epochs 10 --early-stopping-patience 10 --log-radon --grad-clip 1.0 --weight-decay 1e-4)
# training fraction of the RADIM common-period record that ends on the same day as the 70 % of the Jezero station
JP_FRAC=$(cat data/radim_jezero/jezero_period/train_fraction.txt)
run() {  # name seed train_fraction inputs...
  local name=$1 seed=$2 frac=$3; shift 3
  if [[ "${RESUME:-0}" == "1" && -f "$OUT/$name/sequence_scores.csv" ]]; then echo "skip $name"; return; fi
  echo "[$(date '+%F %T')] $name"
  "$PYTHON_BIN" radon_reconstruct_from_window.py --input "$@" --output-dir "$OUT/$name" "${COMMON[@]}" \
    --seed "$seed" --train-fraction "$frac" >"$OUT/logs/$name.log" 2>&1 || echo "FAILED: $name"
}
for s in 42 1 2; do run "radim_jp_s$s" "$s" "$JP_FRAC" data/radim_jezero/jezero_period/bozkov2017jp_part*.csv; done
for s in 1 2; do run "jezero_s$s" "$s" 0.7 data/stations/jezero_part*.csv; done
for s in 1 2; do run "radim_long_s$s" "$s" 0.7 data/radim_jezero/bozkov2017_part*.csv; done
echo "[$(date '+%F %T')] pairs"
"$PYTHON_BIN" -W ignore analysis/probe_comparison.py "$ROOT" || echo "FAILED: analysis/probe_comparison.py"
echo "[$(date '+%F %T')] probe comparison done"
