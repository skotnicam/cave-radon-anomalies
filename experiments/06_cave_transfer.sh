#!/usr/bin/env bash
# Transfer between caves (Section 4.4, Appendix D): LSTM reconstruction trained on the Bozkov reference monitor
# (RADIM, 2017-2026) and applied to Ochtinska (RADIM, 2016-2024), and vice versa; scaling and threshold calibrated
# on the target (--scaling target). Both records come from the same type of instrument.
#
# Usage:  experiments/06_cave_transfer.sh [runs_root]
# Output: <runs_root>/cave_transfer/by_pair/{bozkov2017_to_ochtinska,ochtinska_to_bozkov2017}/
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/cave_transfer"; mkdir -p "$OUT/logs"
# the transfer script reads all records of one directory, so link both caves into one
LINKDIR="$OUT/input"; rm -rf "$LINKDIR"; mkdir -p "$LINKDIR"
for f in data/radim_jezero/bozkov2017_part*.csv data/ochtinska/ochtinska_part*.csv; do ln -s "$(pwd)/$f" "$LINKDIR/$(basename "$f")"; done
for pair in "bozkov2017 ochtinska" "ochtinska bozkov2017"; do
  read -r train test <<< "$pair"; name="${train}_to_${test}"; out="$OUT/by_pair/$name"
  if [[ "${RESUME:-0}" == "1" && -f "$out/sequence_scores.csv" ]]; then echo "skip $name"; continue; fi
  mkdir -p "$out"; echo "[$(date '+%F %T')] $name"
  "$PYTHON_BIN" radon_transfer_reconstruct_radon.py --input-dir "$LINKDIR" \
    --train-stations "$train" --test-station "$test" --output-dir "$out" \
    --window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 \
    --learning-rate 1e-3 --min-epochs 10 --early-stopping-patience 10 \
    --log-radon --scaling target --grad-clip 1.0 --weight-decay 1e-4 \
    > "$OUT/logs/$name.log" 2>&1 || echo "FAILED: $name"
done
echo "[$(date '+%F %T')] cave transfer done"
