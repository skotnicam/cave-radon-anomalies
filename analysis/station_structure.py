#!/usr/bin/env python3
"""Station structure of the Bozkov network: ventilation groups and temperature dependence.

Left: correlation matrix of daily mean log radon between stations (diverging, grey midpoint).
Right: correlation of hourly log radon with outdoor temperature per station, with the
short-term best lag (30-day high-pass) annotated.
Output: <root>/figures/station_structure.{png,pdf} and station_structure.csv
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = os.path.join(ROOT, "figures")
STATIONS = ["prusacka", "za_prahem", "kaple", "blativa", "loupeznicka", "peklo", "jezero"]  # grouped order
NAME = {"prusacka": "Prušácká", "za_prahem": "Za prahem", "loupeznicka": "Loupežnická", "blativa": "Blátivá", "kaple": "Kaple", "peklo": "Peklo", "jezero": "Jezero"}
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold"})


def load(st):
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"data/stations/{st}*.csv"))])
    s["timestamp"] = pd.to_datetime(s.timestamp, utc=True)
    return s.sort_values("timestamp").set_index("timestamp")


hourly = {st: load(st).resample("1h").mean() for st in STATIONS}
daily = pd.concat([np.log1p(h["ARn(Bq/m3)"].clip(lower=0)).resample("1D").mean().rename(st) for st, h in hourly.items()], axis=1)
C = daily.corr()

rows = []
for st, h in hourly.items():
    r = np.log1p(h["ARn(Bq/m3)"].clip(lower=0))
    T = h.temperature_outdoor
    j = pd.concat([r, T], axis=1).dropna()
    raw = j.iloc[:, 0].corr(j.iloc[:, 1])
    rr = r - r.rolling(24 * 30, center=True, min_periods=240).mean()
    TT = T - T.rolling(24 * 30, center=True, min_periods=240).mean()
    best = (0, 0.0)
    for lag in range(0, 73, 2):
        k = pd.concat([rr, TT.shift(lag)], axis=1).dropna()
        c = k.iloc[:, 0].corr(k.iloc[:, 1])
        if abs(c) > abs(best[1]):
            best = (lag, c)
    rows.append(dict(station=st, corr_logRn_Tout=raw, shortterm_best_lag_h=best[0], shortterm_corr=best[1]))
tab = pd.DataFrame(rows).set_index("station")
tab.to_csv(os.path.join(OUT, "station_structure.csv"), float_format="%.3f")
print(C.round(2).to_string())
print(tab.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f4f3ef", BLUE])
im = ax.imshow(C.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(STATIONS)))
ax.set_yticks(range(len(STATIONS)))
ax.set_xticklabels([NAME[s] for s in STATIONS], rotation=45, ha="right")
ax.set_yticklabels([NAME[s] for s in STATIONS])
for i in range(len(STATIONS)):
    for k in range(len(STATIONS)):
        v = C.values[i, k]
        ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if abs(v) > 0.6 else INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_title("Correlation of daily log radon between stations", loc="left")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.outline.set_visible(False)

ax = axes[1]
y = np.arange(len(STATIONS))[::-1]
vals = tab.loc[STATIONS, "corr_logRn_Tout"].values
ax.barh(y, vals, color=[ORANGE if v < 0 else BLUE for v in vals], height=0.6)
for yi, st in zip(y, STATIONS):
    lag, c = tab.loc[st, "shortterm_best_lag_h"], tab.loc[st, "shortterm_corr"]
    v = tab.loc[st, "corr_logRn_Tout"]
    ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"lag {int(lag)} h", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK2)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels([NAME[s] for s in STATIONS])
ax.set_xlim(-1, 1)
ax.set_xlabel("corr(log radon, outdoor temperature), hourly")
ax.set_title("Temperature dependence and short-term lag", loc="left")
ax.grid(axis="x", color=GRID)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"station_structure.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote station_structure")
