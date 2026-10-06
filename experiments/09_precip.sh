#!/usr/bin/env bash
# Sensitivity test of Section 4.3 (Storm Boris): LSTM reconstruction and meteo -> radon retrained with the 24-h and
# 72-h precipitation sums (ERA5-Land) added to the input.
#
# Usage:  experiments/09_precip.sh [runs_root]
# Output: <runs_root>/precip_experiment/by_station/<station>/{reconstruct_radon,meteo_to_radon}/
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/precip_experiment"; mkdir -p "$OUT/logs"
STATIONS=(prusacka za_prahem loupeznicka blativa kaple peklo jezero)
FEAT=(--features "ARn(Bq/m3)" temperature_indoor temperature_outdoor pressure precip_24h precip_72h)
for st in "${STATIONS[@]}"; do
  inputs=(data/stations/${st}*.csv)
  out="$OUT/by_station/$st/reconstruct_radon"
  if [[ "${RESUME:-0}" != "1" || ! -f "$out/sequence_scores.csv" ]]; then
    mkdir -p "$out"; echo "[$(date '+%F %T')] $st reconstruct_radon + precip"
    "$PYTHON_BIN" radon_reconstruct_from_window.py --input "${inputs[@]}" --output-dir "$out" "${FEAT[@]}" \
      --window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 1e-3 \
      --min-epochs 10 --early-stopping-patience 10 --log-radon --grad-clip 1.0 --weight-decay 1e-4 \
      > "$OUT/logs/${st}_reconstruct_radon.log" 2>&1 || echo "FAILED: $st reconstruct_radon"
  fi
  out="$OUT/by_station/$st/meteo_to_radon"
  if [[ "${RESUME:-0}" != "1" || ! -f "$out/sequence_scores.csv" ]]; then
    mkdir -p "$out"; echo "[$(date '+%F %T')] $st meteo_to_radon + precip"
    "$PYTHON_BIN" radon_from_meteo.py --input "${inputs[@]}" --output-dir "$out" "${FEAT[@]}" \
      --window 24 --epochs 50 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 5e-4 \
      --min-epochs 15 --early-stopping-patience 15 --add-temp-diff --log-radon --grad-clip 1.0 --weight-decay 1e-4 \
      > "$OUT/logs/${st}_meteo_to_radon.log" 2>&1 || echo "FAILED: $st meteo_to_radon"
  fi
done
echo "[$(date '+%F %T')] precipitation experiment done"
