#!/usr/bin/env bash
# Capacity test of Section 4.1: meteo -> radon with a larger latent space and hidden width (the canonical run has
# latent 16, hidden 64 and is part of the main suite).
#
# Usage:  experiments/10_meteo_latent.sh [runs_root]
# Output: <runs_root>/meteo_latent/by_station/<station>/{L32_H64,L64_H64,L128_H128}/
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/meteo_latent"; mkdir -p "$OUT/logs"
STATIONS=(prusacka za_prahem kaple blativa loupeznicka peklo jezero)
for st in "${STATIONS[@]}"; do
  inputs=(data/stations/${st}_part*.csv)
  for variant in "L32_H64:32:64" "L64_H64:64:64" "L128_H128:128:128"; do
    IFS=: read -r name latent hidden <<< "$variant"
    out="$OUT/by_station/$st/$name"
    if [[ "${RESUME:-0}" == "1" && -f "$out/sequence_scores.csv" ]]; then echo "skip $st $name"; continue; fi
    mkdir -p "$out"; echo "[$(date '+%F %T')] $st meteo_to_radon $name"
    "$PYTHON_BIN" radon_from_meteo.py --input "${inputs[@]}" --output-dir "$out" \
      --window 24 --epochs 50 --batch-size 32 --hidden-size "$hidden" --latent-size "$latent" \
      --learning-rate 5e-4 --min-epochs 15 --early-stopping-patience 15 --add-temp-diff \
      --log-radon --grad-clip 1.0 --weight-decay 1e-4 >"$OUT/logs/${st}_${name}.log" 2>&1 || echo "FAILED: $st $name"
  done
done
echo "[$(date '+%F %T')] meteo latent done"
