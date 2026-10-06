#!/usr/bin/env python3
"""Two probes at the Jezero site (Jezero station TSR4M vs reference monitor RADIM): decompose the disagreement of
the LSTM flags into training randomness (seed), training period and probe. See experiments/07_probe_comparison.sh.

For every pair of runs, on the common days of 9 Mar 2024 - 20 Mar 2026 (and separately on the training part up to
21 Aug 2025 and the held-out part after it): Jaccard of the daily flags, Spearman correlation of the daily mean
relative error, and the share of 'strong' days (daily max relative error > 2) of one run that the other flags too
(averaged over both directions). Pairs are grouped by what differs between them.
Output: runs/probe_comparison/pairs.csv, summary.md (printed).
"""
import itertools, os
import numpy as np, pandas as pd

import sys
R = sys.argv[1] if len(sys.argv) > 1 else "runs"
PC = f"{R}/probe_comparison"
os.makedirs(PC, exist_ok=True)
RUNS = {
    "jezero_s42": f"{R}/main_suite/by_station/jezero/reconstruct_radon",
    "jezero_s1": f"{PC}/jezero_s1", "jezero_s2": f"{PC}/jezero_s2",
    "radim_jp_s42": f"{PC}/radim_jp_s42", "radim_jp_s1": f"{PC}/radim_jp_s1", "radim_jp_s2": f"{PC}/radim_jp_s2",
    "radim_long_s42": f"{R}/radim_suite/by_station/bozkov2017/reconstruct_radon",
    "radim_long_s1": f"{PC}/radim_long_s1", "radim_long_s2": f"{PC}/radim_long_s2",
}
GROUP = {"jezero": "station Jezero (TSR4M), trained 2024-03..2025-08",
         "radim_jp": "RADIM, trained 2024-03..2025-08 (same period)",
         "radim_long": "RADIM, trained 2017-05..2023-08"}
PERIODS = {"whole overlap": ("2024-03-09", "2026-03-20"), "training part": ("2024-03-09", "2025-08-21"),
           "held-out part": ("2025-08-22", "2026-03-20")}


def load(path):
    d = pd.read_csv(f"{path}/sequence_scores.csv")
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True).dt.tz_localize(None)
    d = d.set_index("timestamp").sort_index()
    return pd.DataFrame({"flag": d.is_anomaly.resample("1D").max(), "err": d.relative_error.resample("1D").mean(),
                         "errmax": d.relative_error.resample("1D").max(),
                         "radon": d["ARn(Bq/m3)"].resample("1D").mean()}).dropna()


D = {k: load(p) for k, p in RUNS.items() if os.path.exists(f"{p}/sequence_scores.csv")}
missing = sorted(set(RUNS) - set(D))


def kind(a, b):
    ga, gb = a.rsplit("_s", 1)[0], b.rsplit("_s", 1)[0]
    if ga == gb:
        return f"seed only: {ga}"
    s = {ga, gb}
    if s == {"jezero", "radim_jp"}:
        return "probe only (same period)"
    if s == {"radim_jp", "radim_long"}:
        return "period only (same probe)"
    return "probe and period"


rows = []
for a, b in itertools.combinations(D, 2):
    for pname, (lo, hi) in PERIODS.items():
        j = pd.concat([D[a].loc[lo:hi], D[b].loc[lo:hi]], axis=1, keys=["a", "b"], join="inner").dropna()
        if len(j) < 20:
            continue
        fa, fb = j["a"]["flag"], j["b"]["flag"]
        inter, union = ((fa == 1) & (fb == 1)).sum(), ((fa == 1) | (fb == 1)).sum()
        sa, sb = j["a"]["errmax"] > 2, j["b"]["errmax"] > 2
        rec = np.nanmean([(fb[sa] == 1).mean() if sa.any() else np.nan, (fa[sb] == 1).mean() if sb.any() else np.nan])
        rows.append(dict(a=a, b=b, kind=kind(a, b), period=pname, days=len(j),
                         jaccard=inter / union if union else np.nan,
                         spearman_err=j["a"]["err"].corr(j["b"]["err"], method="spearman"),
                         strong_days_recovered=rec, rate_a=fa.mean(), rate_b=fb.mean(),
                         radon_corr=j["a"]["radon"].corr(j["b"]["radon"])))
P = pd.DataFrame(rows)
P.to_csv(f"{PC}/pairs.csv", index=False, float_format="%.4f")

order = ["seed only: jezero", "seed only: radim_jp", "seed only: radim_long",
         "probe only (same period)", "period only (same probe)", "probe and period"]
lines = ["# Two probes at the Jezero site: seed, period or probe?\n",
         "Runs: " + ", ".join(f"{k} = {v}" for k, v in GROUP.items()) + ". Seeds 42, 1, 2 each.\n"]
if missing:
    lines.append(f"Missing runs: {missing}\n")
for pname in PERIODS:
    t = (P[P.period == pname].groupby("kind")[["jaccard", "spearman_err", "strong_days_recovered", "days"]]
         .median().reindex([o for o in order if o in set(P.kind)]))
    lines.append(f"\n## {pname} (median over pairs)\n")
    lines.append("| what differs | Jaccard of daily flags | Spearman of daily score | strong days recovered | days |")
    lines.append("|---|---|---|---|---|")
    for k, r in t.iterrows():
        lines.append(f"| {k} | {r.jaccard:.2f} | {r.spearman_err:.2f} | {r.strong_days_recovered:.2f} | {int(r.days)} |")
rates = {k: v.loc["2024-03-09":"2026-03-20", "flag"].mean() for k, v in D.items()}
lines.append("\nDaily flag rate on the overlap: " + ", ".join(f"{k} {v:.3f}" for k, v in rates.items()))
open(f"{PC}/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
