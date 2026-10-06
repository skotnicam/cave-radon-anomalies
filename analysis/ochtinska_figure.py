#!/usr/bin/env python3
"""Ochtinská Aragonite Cave: record, local LSTM flags, the two earthquakes whose Dobrovolsky zone reaches the cave,
and the transfer between Ochtinská and the Bozkov reference monitor.
(a) hourly radon with the flags of the locally trained LSTM reconstruction; shaded = held-out last 30 %;
    dashed = Kahramanmaras Mw 7.8 (2023-02-06) and Zdana ML 5.0 (2023-10-09).
(b) Zdana window: radon, outdoor temperature and pressure, 25 Sep - 10 Nov 2023, with flags.
(c) 2 x 2 transfer (full reference-monitor record): daily-flag Jaccard with the local model of the scored record,
    flagged fraction and median error / threshold (analysis/cave_transfer.py).
Output: <root>/figures/ochtinska.{png,pdf}.  Usage: python analysis/ochtinska_figure.py [runs_root]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = os.path.join(ROOT, "figures")
BLUE, ORANGE, VIOLET, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#4a3aa7", "#8c8c8c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
EQ = {"Kahramanmaraş $M_w$ 7.8": "2023-02-06 01:17", "Ždaňa $M_L$ 5.0": "2023-10-09 18:23"}

s = pd.read_csv(f"{ROOT}/ochtinska_suite/by_cave/ochtinska/reconstruct_radon/sequence_scores.csv")
s["timestamp"] = pd.to_datetime(s.timestamp, utc=True).dt.tz_localize(None)
s = s.set_index("timestamp").sort_index()
raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/ochtinska/ochtinska_part*.csv"))])
raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True).dt.tz_localize(None)
raw = raw.set_index("timestamp").sort_index()
radon = raw["ARn(Bq/m3)"]

fig = plt.figure(figsize=(10, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.6, 1], hspace=0.45, wspace=0.68)

ax = fig.add_subplot(gs[0, :])
# break the line at gaps between segments (> 6 h) instead of bridging them
rgap = radon.copy()
rgap[radon.index.to_series().diff() > pd.Timedelta(hours=6)] = np.nan
ax.plot(rgap.index, rgap.values, color=GREY, lw=0.5)
fl = s[s.is_anomaly == 1]
ax.scatter(fl.index, fl["ARn(Bq/m3)"], s=5, color=BLUE, edgecolors="none", zorder=3, label="flagged (local LSTM)")
k = int(len(s) * 0.7)
ax.axvspan(s.index[k], s.index[-1], color="#f2f2f2", zorder=0)
ax.text(s.index[k], 0.97, "  held-out 30 %", transform=ax.get_xaxis_transform(), fontsize=9, color=INK2, va="top")
for (name, t), (ha, y) in zip(EQ.items(), [("right", 0.03), ("left", 0.13)]):
    ax.axvline(pd.Timestamp(t), color=ORANGE, lw=1.1, ls="--")
    ax.text(pd.Timestamp(t), y, (name + " ") if ha == "right" else (" " + name), transform=ax.get_xaxis_transform(),
            fontsize=9, color=ORANGE, ha=ha, va="bottom")
ax.set_yscale("log"); ax.set_ylabel("radon (Bq/m³)")
ax.set_title(f"(a) Ochtinská Aragonite Cave, hourly radon, {s.is_anomaly.mean() * 100:.1f} % flagged", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

a, b = "2023-09-25", "2023-11-10"
sub = gs[1, 0].subgridspec(2, 1, height_ratios=[1.3, 1], hspace=0.08)
ax = fig.add_subplot(sub[0])
w = radon.loc[a:b]
ax.plot(w.index, w.values, color=GREY, lw=0.8)
f = s.loc[a:b]; f = f[f.is_anomaly == 1]
ax.scatter(f.index, f["ARn(Bq/m3)"], s=7, color=BLUE, zorder=3)
ax.set_ylabel("radon\n(Bq/m³)")
ax.set_title("(b) Around the Ždaňa earthquake", loc="left")
ax.tick_params(labelbottom=False)
axm = fig.add_subplot(sub[1], sharex=ax)
axm.plot(raw.loc[a:b, "temperature_outdoor"].resample("3h").mean(), color=VIOLET, lw=1.2)
axm.set_ylabel("outdoor\nT (°C)", color=VIOLET); axm.tick_params(axis="y", colors=VIOLET)
axp = axm.twinx()
axp.plot(raw.loc[a:b, "pressure"].resample("3h").mean(), color=INK, lw=1, ls=":")
axp.set_ylabel("pressure\n(hPa)")
for x in (ax, axm):
    x.axvline(pd.Timestamp(EQ["Ždaňa $M_L$ 5.0"]), color=ORANGE, lw=1.1, ls="--")
    x.spines["top"].set_visible(False)
axm.xaxis.set_major_locator(mdates.DayLocator(interval=10)); axm.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

T = lambda n: pd.read_csv(os.path.join(OUT, f"transfer2017_{n}.csv"), index_col=0).loc[["bozkov2017", "ochtinska"], ["bozkov2017", "ochtinska"]]
rate, med, jac = T("anomaly_rate"), T("median_relative_error"), T("jaccard")
ax = fig.add_subplot(gs[1, 1])
ax.imshow(jac.values, cmap="Blues", vmin=0, vmax=1)
lab = ["Bozkov\n(RADIM,\n2017–2026)", "Ochtinská"]
ax.set_xticks([0, 1]); ax.set_xticklabels(lab); ax.set_yticks([0, 1]); ax.set_yticklabels(lab)
ax.set_xlabel("scored record"); ax.set_ylabel("trained on")
for i in range(2):
    for j in range(2):
        v = jac.values[i, j]
        ax.text(j, i, f"J = {v:.2f}\n{rate.values[i, j] * 100:.1f} %\nmed. {med.values[i, j]:.2f}", ha="center", va="center",
                fontsize=9.5, color="white" if v > 0.6 else INK, fontweight="bold" if i == j else "normal")
ax.set_title("(c) Transfer between caves", loc="left")
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ochtinska.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote ochtinska; flagged", round(s.is_anomaly.mean(), 4), "train ends", s.index[k - 1].date())
