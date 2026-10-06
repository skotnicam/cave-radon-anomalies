#!/usr/bin/env bash
# Leave-event-out control on the RADIM record (Section 4.3): the LSTM reconstruction model is trained on the record
# without +-60 days around the five earthquakes whose preparation zone reached the cave, and scores the +-30 days
# around them out of sample. Three seeds. The training record is rebuilt here from data/radim_jezero.
#
# Usage:  experiments/08_radim_leaveout.sh [runs_root]
# Output: <runs_root>/radim_leaveout/s<seed>/ (and the derived training/event segments in <runs_root>/radim_leaveout/data)
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="$ROOT/radim_leaveout"; DATA="$OUT/data"; mkdir -p "$OUT/logs" "$DATA"

"$PYTHON_BIN" -W ignore - "$DATA" <<'PY'
import glob, os, sys, pandas as pd
out = sys.argv[1]
ev = ["2017-12-26", "2019-01-29", "2020-07-08", "2020-12-29", "2023-02-06"]   # Lubin x3, Petrinja, Kahramanmaras
for f in glob.glob(out + "/*.csv"): os.remove(f)
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/radim_jezero/bozkov2017_part*.csv"))])
d["t"] = pd.to_datetime(d.timestamp)
excl = pd.Series(False, index=d.index); test = pd.Series(-1, index=d.index)
for i, e in enumerate(ev):
    t = pd.Timestamp(e)
    excl |= (d.t - t).abs() <= pd.Timedelta(days=60)
    test[(d.t - t).abs() <= pd.Timedelta(days=30)] = i
def write(part, prefix):
    new = (part.t.diff() > pd.Timedelta(hours=6)).cumsum()   # segments: gaps > 6 h
    n = 0
    for _, g in part.groupby(new):
        if len(g) >= 500 or prefix == "radimevent":
            if len(g) >= 48:
                n += 1; g.drop(columns="t").to_csv(f"{out}/{prefix}_{len(glob.glob(out + '/' + prefix + '*')) + 1:02d}.csv", index=False)
    return n
print("training rows", len(d[~excl]), "parts", write(d[~excl], "radimtrain"))
for i in range(len(ev)):
    print("event", ev[i], "rows", int((test == i).sum()), "parts", write(d[test == i], "radimevent"))
PY

for s in 42 1 2; do
  if [[ "${RESUME:-0}" == "1" && -f "$OUT/s$s/sequence_scores.csv" ]]; then echo "skip s$s"; continue; fi
  echo "[$(date '+%F %T')] leave-out seed $s"
  # --scaling source: the event windows are scored with the scaling and threshold of the training record
  "$PYTHON_BIN" radon_transfer_reconstruct_radon.py --input-dir "$DATA" --train-stations radimtrain \
    --test-station radimevent --output-dir "$OUT/s$s" --scaling source --seed "$s" \
    --window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 1e-3 \
    --min-epochs 10 --early-stopping-patience 10 --log-radon --grad-clip 1.0 --weight-decay 1e-4 \
    > "$OUT/logs/s$s.log" 2>&1 || echo "FAILED: s$s"
done
echo "[$(date '+%F %T')] leave-out done"
