#!/usr/bin/env python3
"""Table 4 of the paper: Jaccard overlap of the daily flags of the models trained on three stations
(experiments/05_multistation.sh) with those of the locally trained model of each scored station, plus the best
single-station donor from the station-transfer matrix (analysis/station_transfer.py).
Output: <root>/figures/multistation_jaccard.csv.  Usage: python analysis/multistation_table.py [runs_root]
"""
import os, sys
import numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = f"{ROOT}/figures"; os.makedirs(OUT, exist_ok=True)
STATIONS = ["prusacka", "za_prahem", "kaple", "blativa", "loupeznicka", "peklo", "jezero"]
COMBOS = {"kaple_zaprahem_blativa": ["kaple", "za_prahem", "blativa"], "kaple_prusacka_loupeznicka": ["kaple", "prusacka", "loupeznicka"]}


def daily_flags(path):
    d = pd.read_csv(path); d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    return d.set_index("timestamp").sort_index().is_anomaly.resample("1D").max()


def jaccard(a, b):
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    inter, union = ((j.a == 1) & (j.b == 1)).sum(), ((j.a == 1) | (j.b == 1)).sum()
    return inter / union if union else np.nan


local = {st: daily_flags(f"{ROOT}/main_suite/by_station/{st}/reconstruct_radon/sequence_scores.csv") for st in STATIONS}
rows = {}
for combo in COMBOS:
    rows[combo] = {}
    for te in STATIONS:
        p = f"{ROOT}/multistation_transfer/by_pair/{combo}_to_{te}/sequence_scores.csv"
        rows[combo][te] = jaccard(local[te], daily_flags(p)) if os.path.exists(p) else np.nan
T = pd.DataFrame(rows).T[STATIONS]
st_path = f"{OUT}/station_transfer_jaccard.csv"
if os.path.exists(st_path):
    M = pd.read_csv(st_path, index_col=0)
    T.loc["best single station"] = [M.loc[[s for s in STATIONS if s != te], te].max() for te in STATIONS]
T.to_csv(f"{OUT}/multistation_jaccard.csv", float_format="%.3f")
print("daily-flag Jaccard with the local model (asterisk in the paper: station in the training set)")
print(T.round(2).to_string())
