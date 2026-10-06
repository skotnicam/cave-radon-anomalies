#!/usr/bin/env python3
"""Storm Boris with and without precipitation as a model input (Section 4.3). For the LSTM reconstruction and the
meteo -> radon model, the network mean of the daily error percentile (each station's daily mean relative error
ranked within its own record, averaged over stations with data) over the storm week (12-18 September 2024), for the canonical runs
(main suite) and the runs retrained with the 24-h and 72-h precipitation sums (experiments/09_precip.sh).
Output: <root>/figures/precip_boris_summary.csv.  Usage: python analysis/precip_summary.py [runs_root]
"""
import os, sys
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = f"{ROOT}/figures"; os.makedirs(OUT, exist_ok=True)
STATIONS = ["prusacka", "za_prahem", "loupeznicka", "blativa", "kaple", "peklo", "jezero"]
STORM_WEEK = slice(pd.Timestamp("2024-09-12", tz="UTC"), pd.Timestamp("2024-09-18", tz="UTC"))  # as in rain_error.py
WEEK_BEFORE = slice(pd.Timestamp("2024-09-05", tz="UTC"), pd.Timestamp("2024-09-11", tz="UTC"))


def network_percentile(paths):
    err = {}
    for st, p in paths.items():
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p); d["timestamp"] = pd.to_datetime(d.timestamp, utc=True, errors="coerce")
        s = d.dropna(subset=["timestamp"]).set_index("timestamp").sort_index().relative_error.resample("1D").mean()
        err[st] = s.rank(pct=True)
    err = pd.DataFrame(err)
    return err.mean(axis=1).where(err.notna().sum(axis=1) >= 3)


rows = []
for method in ["reconstruct_radon", "meteo_to_radon"]:
    for name, base in [("canonical", f"{ROOT}/main_suite/by_station"), ("with precipitation", f"{ROOT}/precip_experiment/by_station")]:
        net = network_percentile({st: f"{base}/{st}/{method}/sequence_scores.csv" for st in STATIONS})
        if net.notna().sum() == 0:
            continue
        rows.append(dict(method=method, model=name, week_before=net.loc[WEEK_BEFORE].mean(), storm_week=net.loc[STORM_WEEK].mean()))
T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/precip_boris_summary.csv", index=False, float_format="%.3f")
print("network mean error percentile, Storm Boris week (12-18 Sep 2024) and the week before")
print(T.round(3).to_string(index=False))
