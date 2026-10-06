#!/usr/bin/env bash
# Models trained on three stations together and applied to every station (Table 4 of the paper).
#
# Usage:  experiments/05_multistation.sh [runs_root]
# Output: <runs_root>/multistation_transfer/by_pair/<combo>_to_<test>/
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/multistation_transfer"; mkdir -p "$OUT/logs"
STATIONS=(prusacka za_prahem kaple blativa loupeznicka peklo jezero)
declare -A COMBOS=( [kaple_zaprahem_blativa]="kaple za_prahem blativa" [kaple_prusacka_loupeznicka]="kaple prusacka loupeznicka" )
for combo in kaple_zaprahem_blativa kaple_prusacka_loupeznicka; do
  for test in "${STATIONS[@]}"; do
    out="$OUT/by_pair/${combo}_to_${test}"
    if [[ "${RESUME:-0}" == "1" && -f "$out/sequence_scores.csv" ]]; then echo "skip $combo -> $test"; continue; fi
    mkdir -p "$out"; echo "[$(date '+%F %T')] $combo -> $test"
    # shellcheck disable=SC2086
    "$PYTHON_BIN" radon_transfer_reconstruct_radon.py --input-dir data/stations \
      --train-stations ${COMBOS[$combo]} --test-station "$test" --output-dir "$out" \
      --window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 \
      --learning-rate 1e-3 --min-epochs 10 --early-stopping-patience 10 \
      --log-radon --scaling target --grad-clip 1.0 --weight-decay 1e-4 \
      > "$OUT/logs/${combo}_to_${test}.log" 2>&1 || echo "FAILED: $combo -> $test"
  done
done
echo "[$(date '+%F %T')] multistation transfer done"
