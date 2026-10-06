#!/usr/bin/env python3
"""Intra-cave transfer matrices for the Bozkov network (train on one station, score another).

Three 7x7 matrices (rows = training station, columns = scored station):
  anomaly rate on the scored station; median error / threshold; Jaccard overlap of the daily
  flags with the locally trained model of the scored station. Diagonal = locally trained model
  (runs/main_suite, reconstruct_radon).
Output: <root>/figures/station_transfer.{png,pdf} + station_transfer_{rate,medrel,jaccard}.csv
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = os.path.join(ROOT, "figures")
STATIONS = ["prusacka", "za_prahem", "kaple", "blativa", "loupeznicka", "peklo", "jezero"]
NAME = {"prusacka": "Prušácká", "za_prahem": "Za prahem", "loupeznicka": "Loupežnická", "blativa": "Blátivá", "kaple": "Kaple", "peklo": "Peklo", "jezero": "Jezero"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold"})


def read(path):
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    return d.set_index("timestamp").sort_index()


local = {st: read(f"{ROOT}/main_suite/by_station/{st}/reconstruct_radon/sequence_scores.csv") for st in STATIONS}
rate = pd.DataFrame(index=STATIONS, columns=STATIONS, dtype=float)
medrel = rate.copy()
jac = rate.copy()
for tr in STATIONS:
    for te in STATIONS:
        if tr == te:
            d = local[te]
        else:
            p = f"{ROOT}/station_transfer/by_pair/{tr}_to_{te}/sequence_scores.csv"
            if not os.path.exists(p):
                continue
            d = read(p)
        rate.loc[tr, te] = d.is_anomaly.mean()
        medrel.loc[tr, te] = d.relative_error.median()
        a = local[te].is_anomaly.resample("1D").max()
        b = d.is_anomaly.resample("1D").max()
        j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        inter = ((j.a == 1) & (j.b == 1)).sum()
        union = ((j.a == 1) | (j.b == 1)).sum()
        jac.loc[tr, te] = inter / union if union else np.nan
for name, M in [("rate", rate), ("medrel", medrel), ("jaccard", jac)]:
    M.to_csv(os.path.join(OUT, f"station_transfer_{name}.csv"), float_format="%.4f")
    print(f"\n{name}\n", M.astype(float).round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
panels = [
    (rate, "Anomaly rate on the scored station", "{:.1%}", 0, None),
    (medrel, "Median error / threshold", "{:.2f}", 0, None),
    (jac, "Daily-flag overlap with the local model (Jaccard)", "{:.2f}", 0, 1),
]
for ax, (M, title, fmt, vmin, vmax) in zip(axes, panels):
    vals = M.values.astype(float)
    vmax_ = vmax if vmax is not None else np.nanmax(vals)
    im = ax.imshow(vals, cmap="Blues", vmin=vmin, vmax=vmax_)
    ax.set_xticks(range(len(STATIONS)))
    ax.set_yticks(range(len(STATIONS)))
    ax.set_xticklabels([NAME[s] for s in STATIONS], rotation=45, ha="right")
    ax.set_yticklabels([NAME[s] for s in STATIONS] if ax is axes[0] else [])
    ax.set_xlabel("scored station")
    if ax is axes[0]:
        ax.set_ylabel("training station")
    ax.set_title(title, loc="left")
    for i in range(len(STATIONS)):
        for k in range(len(STATIONS)):
            v = vals[i, k]
            if np.isfinite(v):
                ax.text(k, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 * vmax_ else INK, fontweight="bold" if i == k else "normal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
fig.suptitle("LSTM radon reconstruction trained on one Bozkov station and applied to another (bold diagonal = local model)",
             x=0.01, ha="left", color=INK2, fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"station_transfer.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote station_transfer")
