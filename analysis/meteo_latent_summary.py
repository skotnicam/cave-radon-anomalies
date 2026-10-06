#!/usr/bin/env python3
"""Capacity test of the meteo -> radon model (Section 4.1): final training loss, best validation loss and their ratio
for the canonical run (latent 16, hidden 64; main suite) and the larger variants of experiments/10_meteo_latent.sh,
median over stations.
Output: <root>/figures/meteo_latent_summary.csv.  Usage: python analysis/meteo_latent_summary.py [runs_root]
"""
import os, sys
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = f"{ROOT}/figures"; os.makedirs(OUT, exist_ok=True)
STATIONS = ["prusacka", "za_prahem", "kaple", "blativa", "loupeznicka", "peklo", "jezero"]
rows = []
for st in STATIONS:
    variants = [("L16_H64 (canonical)", f"{ROOT}/main_suite/by_station/{st}/meteo_to_radon")] + \
               [(v, f"{ROOT}/meteo_latent/by_station/{st}/{v}") for v in ("L32_H64", "L64_H64", "L128_H128")]
    for name, path in variants:
        h = f"{path}/training_history.csv"; s = f"{path}/sequence_scores.csv"
        if not (os.path.exists(h) and os.path.exists(s)):
            continue
        hh = pd.read_csv(h); d = pd.read_csv(s)
        rows.append(dict(station=st, variant=name, epochs=len(hh), final_train=hh.train_loss.iloc[-1], best_val=hh.val_loss.min(),
                         val_over_train=hh.val_loss.iloc[-1] / hh.train_loss.iloc[-1], anomaly_rate=d.is_anomaly.mean()))
A = pd.DataFrame(rows)
A.to_csv(f"{OUT}/meteo_latent_per_station.csv", index=False, float_format="%.4f")
M = A.groupby("variant")[["epochs", "final_train", "best_val", "val_over_train", "anomaly_rate"]].median()
M.to_csv(f"{OUT}/meteo_latent_summary.csv", float_format="%.4f")
print("median over stations"); print(M.round(3).to_string())
