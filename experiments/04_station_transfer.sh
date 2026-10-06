#!/usr/bin/env bash
# Transfer between stations (Section 4.4): the LSTM reconstruction model trained on one station and applied to
# another without parameter update; input scaling and threshold calibrated on the first 70 % of the target
# (--scaling target). 42 ordered pairs.
#
# Usage:  experiments/04_station_transfer.sh [runs_root]
# Output: <runs_root>/station_transfer/by_pair/<train>_to_<test>/
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/station_transfer"; mkdir -p "$OUT/logs"
STATIONS=(prusacka za_prahem loupeznicka blativa kaple peklo jezero)
for train in "${STATIONS[@]}"; do
  for test in "${STATIONS[@]}"; do
    [[ "$train" == "$test" ]] && continue
    pair="${train}_to_${test}"; out="$OUT/by_pair/$pair"
    if [[ "${RESUME:-0}" == "1" && -f "$out/sequence_scores.csv" ]]; then echo "skip $pair"; continue; fi
    mkdir -p "$out"; echo "[$(date '+%F %T')] $pair"
    "$PYTHON_BIN" radon_transfer_reconstruct_radon.py --input-dir data/stations \
      --train-stations "$train" --test-station "$test" --output-dir "$out" \
      --window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 \
      --learning-rate 1e-3 --min-epochs 10 --early-stopping-patience 10 \
      --log-radon --scaling target --grad-clip 1.0 --weight-decay 1e-4 \
      > "$OUT/logs/$pair.log" 2>&1 || echo "FAILED: $pair"
  done
done
echo "[$(date '+%F %T')] station transfer done"
