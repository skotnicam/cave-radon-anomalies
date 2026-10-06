#!/usr/bin/env bash
# Ochtinska Aragonite Cave (RADIM, 2016-2024): LSTM and Transformer reconstruction and MMLR (Appendix D).
#
# Usage:  experiments/03_ochtinska.sh [runs_root]
# Output: <runs_root>/ochtinska_suite/by_cave/ochtinska/<method>/
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/ochtinska_suite/by_cave/ochtinska"; LOG="$ROOT/ochtinska_suite/logs"; mkdir -p "$LOG"
inputs=(data/ochtinska/ochtinska_part*.csv)
NEURAL=(--window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 1e-3
        --min-epochs 10 --early-stopping-patience 10 --log-radon --grad-clip 1.0 --weight-decay 1e-4)

run() {  # method script args...
  local m=$1 script=$2; shift 2
  if [[ "${RESUME:-0}" == "1" && -f "$OUT/$m/sequence_scores.csv" ]]; then echo "skip ochtinska/$m"; return; fi
  mkdir -p "$OUT/$m"; echo "[$(date '+%F %T')] ochtinska $m"
  "$PYTHON_BIN" "$script" --input "${inputs[@]}" --output-dir "$OUT/$m" "$@" >"$LOG/$m.log" 2>&1 || echo "FAILED: ochtinska/$m"
}
run reconstruct_radon radon_reconstruct_from_window.py "${NEURAL[@]}"
run transformer_reconstruct_radon radon_transformer_reconstruct_from_window.py "${NEURAL[@]}" --transformer-nhead 4 --transformer-dim-feedforward 128
run mmlr radon_mmlr.py --add-time-features --log-radon
echo "[$(date '+%F %T')] ochtinska done"
