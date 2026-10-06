#!/usr/bin/env bash
# Main suite: the ten methods on every station, one configuration (Table 2 of the paper).
#
# Usage:  experiments/01_main_suite.sh [runs_root]        (default runs/)
# Output: <runs_root>/main_suite/by_station/<station>/<method>/sequence_scores.csv (+ training_history.csv, plots)
#         <runs_root>/main_suite/logs/<station>/<method>.log
# Env:    STATIONS   space-separated station prefixes (default: the seven Bozkov stations)
#         INPUT_DIR  directory with <station>*.csv segments (default data/stations)
#         METHODS    space-separated subset of the ten methods (default: all)
#         SUITE_OUT  output directory (default <runs_root>/main_suite)
#         RESUME=1   skip runs whose sequence_scores.csv exists
#         PYTHON_BIN python executable (default python)
# The whole suite (70 runs) takes about 75 minutes on a laptop CPU.
set -uo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:-runs}"
OUT="${SUITE_OUT:-$ROOT/main_suite}"
INPUT_DIR="${INPUT_DIR:-data/stations}"
PYTHON_BIN="${PYTHON_BIN:-python}"
read -r -a STATIONS <<< "${STATIONS:-prusacka za_prahem loupeznicka blativa kaple peklo jezero}"
read -r -a METHODS <<< "${METHODS:-reconstruct_radon reconstruct_delayed_temp transformer_reconstruct_radon transformer_reconstruct_delayed_temp conv_reconstruct_radon forecast conv_forecast meteo_to_radon mmlr emd_svr}"

# shared settings of the neural models (Section 3.3): 24-h window, log1p radon, Adam 1e-3, batch 32,
# weight decay 1e-4, gradient clipping 1, at most 40 epochs, early stopping with patience 10 after 10 epochs
NEURAL=(--window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 1e-3
        --min-epochs 10 --early-stopping-patience 10 --log-radon --grad-clip 1.0 --weight-decay 1e-4)
FORECAST=(--window 24 --forecast-horizon 3 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 32
          --learning-rate 1e-3 --min-epochs 10 --early-stopping-patience 10 --log-radon --grad-clip 1.0 --weight-decay 1e-4)

run() {  # station method script args...
  local station=$1 method=$2 script=$3; shift 3
  local out="$OUT/by_station/$station/$method" log="$OUT/logs/$station/$method.log"
  if [[ "${RESUME:-0}" == "1" && -f "$out/sequence_scores.csv" ]]; then echo "skip $station/$method"; return; fi
  mkdir -p "$out" "$(dirname "$log")"
  shopt -s nullglob; local inputs=("$INPUT_DIR/$station"*.csv); shopt -u nullglob
  if [[ ${#inputs[@]} -eq 0 ]]; then echo "no input for $station in $INPUT_DIR" >&2; return 1; fi
  echo "[$(date '+%F %T')] $station $method"
  "$PYTHON_BIN" "$script" --input "${inputs[@]}" --output-dir "$out" "$@" >"$log" 2>&1 || echo "FAILED: $station/$method (see $log)"
}

for st in "${STATIONS[@]}"; do
  for m in "${METHODS[@]}"; do
    case "$m" in
      reconstruct_radon)                    run "$st" "$m" radon_reconstruct_from_window.py "${NEURAL[@]}" ;;
      reconstruct_delayed_temp)             run "$st" "$m" radon_reconstruct_delayed_temp.py "${NEURAL[@]}" --temp-outdoor-lag-hours 48 --replace-temp-outdoor-with-lag ;;
      transformer_reconstruct_radon)        run "$st" "$m" radon_transformer_reconstruct_from_window.py "${NEURAL[@]}" --transformer-nhead 4 --transformer-dim-feedforward 128 ;;
      transformer_reconstruct_delayed_temp) run "$st" "$m" radon_transformer_reconstruct_delayed_temp.py "${NEURAL[@]}" --transformer-nhead 4 --transformer-dim-feedforward 128 --temp-outdoor-lag-hours 48 --replace-temp-outdoor-with-lag ;;
      conv_reconstruct_radon)               run "$st" "$m" radon_conv_reconstruct_from_window.py "${NEURAL[@]}" --conv-kernel-size 5 --conv-temporal-bottleneck ;;
      forecast)                             run "$st" "$m" radon_forecast.py "${FORECAST[@]}" ;;
      conv_forecast)                        run "$st" "$m" radon_conv_forecast.py "${FORECAST[@]}" --conv-kernel-size 5 --conv-temporal-bottleneck ;;
      meteo_to_radon)                       run "$st" "$m" radon_from_meteo.py --window 24 --epochs 50 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 5e-4 --min-epochs 15 --early-stopping-patience 15 --add-temp-diff --log-radon --grad-clip 1.0 --weight-decay 1e-4 ;;
      mmlr)                                 run "$st" "$m" radon_mmlr.py --add-time-features --log-radon ;;
      emd_svr)                              run "$st" "$m" radon_emd_svr.py ;;
    esac
  done
done
echo "[$(date '+%F %T')] main suite done: $OUT"
