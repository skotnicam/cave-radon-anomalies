#!/usr/bin/env python3
"""Agreement of every method with the LSTM radon reconstruction, beyond the plain Jaccard index.
Per station and method: Jaccard of hourly and daily flags, the Jaccard expected for independent flags at the
same rates (chance), the share of reference hours scored > 2x threshold that the other method also flags
(strong events), the same for 1.0-1.3x threshold (marginal), and the Spearman correlation of the two scores.
Output: <root>/figures/method_agreement_strength.csv (per station) + printed medians.
Usage: python analysis/method_agreement.py [runs_root]
"""
import os, sys
import numpy as np, pandas as pd
ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
BASE = f"{ROOT}/main_suite/by_station"
ST = ["prusacka", "za_prahem", "kaple", "blativa", "loupeznicka", "peklo", "jezero"]
M = ["reconstruct_delayed_temp", "transformer_reconstruct_radon", "transformer_reconstruct_delayed_temp",
     "conv_reconstruct_radon", "forecast", "conv_forecast", "meteo_to_radon", "mmlr", "emd_svr"]


def read(st, m):
    d = pd.read_csv(f"{BASE}/{st}/{m}/sequence_scores.csv"); d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    return d.set_index("timestamp").sort_index()


def jac(a, b):
    i = ((a == 1) & (b == 1)).sum(); u = ((a == 1) | (b == 1)).sum(); return i / u if u else np.nan


rows = []
for st in ST:
    ref = read(st, "reconstruct_radon")
    for m in M:
        o = read(st, m)
        j = pd.concat([ref.relative_error, o.relative_error, ref.is_anomaly, o.is_anomaly], axis=1,
                      keys=["ra", "rb", "fa", "fb"], join="inner")
        p, q = j.fa.mean(), j.fb.mean()
        rows.append(dict(station=st, method=m, jaccard_hourly=jac(j.fa, j.fb),
                         jaccard_daily=jac(j.fa.resample("1D").max(), j.fb.resample("1D").max()),
                         jaccard_chance=p * q / (p + q - p * q),
                         strong_recovered=(j[j.ra > 2].fb == 1).mean(),
                         marginal_recovered=(j[(j.ra > 1) & (j.ra <= 1.3)].fb == 1).mean(),
                         spearman_scores=j.ra.corr(j.rb, method="spearman")))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, "figures", "method_agreement_strength.csv"), index=False, float_format="%.4f")
print(R.groupby("method").median(numeric_only=True).loc[M].round(2).to_string())
