#!/usr/bin/env python3
"""Two probes at the Jezero site: TSR4M station vs RADIM reference monitor (figure for the paper).
(a) daily mean radon of both probes, March 2024 - March 2026;
(b) daily maximum relative error of the LSTM reconstruction trained on the same period at both probes
    (station: canonical run; RADIM: probe_comparison/radim_jp_s42), threshold = 1, strong events labelled;
(c) agreement of the daily flags by what differs between two runs (from probe_comparison/pairs.csv, whole
    overlap): Jaccard and share of strong days (> 2x threshold) recovered, median over pairs.
Output: <root>/figures/two_probes.{png,pdf}.  Usage: python analysis/two_probes_figure.py [runs_root]
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

R = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = f"{R}/figures"
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#c9c9c9", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.titleweight": "bold"})
LO, HI = "2024-03-09", "2026-03-20"


def raw(files):
    d = pd.concat([pd.read_csv(f) for f in files])
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True).dt.tz_localize(None)
    return d.set_index("timestamp")["ARn(Bq/m3)"].resample("1D").mean().loc[LO:HI]


def scores(p):
    d = pd.read_csv(f"{p}/sequence_scores.csv")
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True).dt.tz_localize(None)
    d = d.set_index("timestamp")
    return d.relative_error.resample("1D").max().loc[LO:HI], d.is_anomaly.resample("1D").max().loc[LO:HI]


tsr = raw(sorted(glob.glob("data/stations/jezero_part*.csv")))
rad = raw(sorted(glob.glob("data/radim_jezero/jezero_period/*.csv")))
e_t, f_t = scores(f"{R}/main_suite/by_station/jezero/reconstruct_radon")
e_r, f_r = scores(f"{R}/probe_comparison/radim_jp_s42")
P = pd.read_csv(f"{R}/probe_comparison/pairs.csv")
P = P[P.period == "whole overlap"]

fig = plt.figure(figsize=(10, 7.6))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.3, 1.05], hspace=0.55)
ax = fig.add_subplot(gs[0])
ax.plot(tsr.index, tsr, color=BLUE, lw=1.2, label="station Jezero (TSR4M)")
ax.plot(rad.index, rad, color=ORANGE, lw=1.2, label="reference monitor (RADIM)")
ax.set_yscale("log"); ax.set_ylabel("radon (Bq/m³)"); ax.set_ylim(300, 4e4)
J = pd.concat([tsr, rad], axis=1).dropna()
ax.set_title(f"(a) Daily mean radon at the Jezero site; correlation {J.iloc[:, 0].corr(J.iloc[:, 1]):.3f}, "
             f"level ratio {np.median(J.iloc[:, 1] / J.iloc[:, 0]):.2f}", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)

ax = fig.add_subplot(gs[1], sharex=ax)
ax.plot(e_t.index, e_t, color=BLUE, lw=1, label="TSR4M model")
ax.plot(e_r.index, e_r, color=ORANGE, lw=1, label="RADIM model")
ax.axhline(1, color=INK2, lw=0.8, ls="--")
ax.set_yscale("log"); ax.set_ylabel("daily max error / threshold")
for d, name, y, ha in [("2024-09-14", "Boris ", 0.97, "right"), ("2025-10-13", "12–15 Oct 2025 ", 0.97, "right"), ("2025-11-22", " 22 Nov 2025", 0.97, "left")]:
    t = pd.Timestamp(d)
    ax.axvline(t, color=GREY, lw=0.8, zorder=0)
    ax.text(t, y, name, transform=ax.get_xaxis_transform(), fontsize=9, color=INK2, va="top", ha=ha)
ax.set_title("(b) LSTM reconstruction trained on the same period (to 21 Aug 2025) at both probes", loc="left")
ax.set_ylim(0.1, 3e3)
ax.legend(frameon=False, fontsize=9, loc="center left", ncol=2)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
for a in fig.axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)

ax = fig.add_subplot(gs[2])
order = [("seed only: jezero", "seed only,\nTSR4M"), ("seed only: radim_jp", "seed only,\nRADIM"),
         ("period only (same probe)", "training period\nonly"), ("probe only (same period)", "probe only"),
         ("probe and period", "probe and\nperiod")]
med = P.groupby("kind")[["jaccard", "strong_days_recovered"]].median()
x = np.arange(len(order)); w = 0.36
ax.bar(x - w / 2, [med.loc[k, "jaccard"] for k, _ in order], w, color=BLUE, label="Jaccard of daily flags")
ax.bar(x + w / 2, [med.loc[k, "strong_days_recovered"] for k, _ in order], w, color=ORANGE, label="strong days recovered")
for i, (k, _) in enumerate(order):
    ax.text(i - w / 2, med.loc[k, "jaccard"] + 0.02, f"{med.loc[k, 'jaccard']:.2f}", ha="center", fontsize=8.5)
    ax.text(i + w / 2, med.loc[k, "strong_days_recovered"] + 0.02, f"{med.loc[k, 'strong_days_recovered']:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=9.5)
ax.set_ylim(0, 1.15); ax.set_ylabel("agreement")
ax.set_title("(c) Agreement between two runs, by what differs between them (median over seed pairs)", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/two_probes.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote two_probes")
