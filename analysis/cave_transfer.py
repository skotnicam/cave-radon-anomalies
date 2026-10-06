#!/usr/bin/env python3
"""Cross-cave transfer between the full Bozkov reference-monitor record (bozkov2017, 2017-2026) and Ochtinska.
Local models: radim_suite (reconstruct_radon) and ochtinska_suite/by_cave/ochtinska.
Transfers: cave_transfer/by_pair/{bozkov2017_to_ochtinska, ochtinska_to_bozkov2017} (--scaling target).
Output: <root>/figures/transfer2017_{anomaly_rate,median_relative_error,jaccard}.csv (rows: training, cols: scored).
"""
import os, sys
import numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "runs"
LOCAL = {"bozkov2017": f"{R}/radim_suite/by_station/bozkov2017/reconstruct_radon",
         "ochtinska": f"{R}/ochtinska_suite/by_cave/ochtinska/reconstruct_radon"}
C = list(LOCAL)


def read(p):
    d = pd.read_csv(f"{p}/sequence_scores.csv"); d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    return d.set_index("timestamp").sort_index()


rate, med, jac = (pd.DataFrame(index=C, columns=C, dtype=float) for _ in range(3))
for tr in C:
    for te in C:
        d = read(LOCAL[te] if tr == te else f"{R}/cave_transfer/by_pair/{tr}_to_{te}")
        loc = read(LOCAL[te])
        rate.loc[tr, te], med.loc[tr, te] = d.is_anomaly.mean(), d.relative_error.median()
        j = pd.concat([loc.is_anomaly.resample("1D").max(), d.is_anomaly.resample("1D").max()], axis=1, keys=["a", "b"]).dropna()
        u = ((j.a == 1) | (j.b == 1)).sum(); jac.loc[tr, te] = ((j.a == 1) & (j.b == 1)).sum() / u if u else np.nan
for n, M in [("anomaly_rate", rate), ("median_relative_error", med), ("jaccard", jac)]:
    M.to_csv(f"{R}/figures/transfer2017_{n}.csv", float_format="%.4f")
    print(n); print(M.round(3).to_string(), "\n")
