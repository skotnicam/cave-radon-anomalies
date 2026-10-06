#!/usr/bin/env python3
"""Continuous reconstruction error vs precipitation (Storm Boris and other heavy-rain events).

Score = daily mean of relative_error (error / threshold) of the LSTM reconstruct_radon model,
converted to a percentile rank within each station's own record so stations are comparable.
Precipitation and pressure: open-meteo ERA5-Land archive for Bozkov (data/meteo/openmeteo_bozkov_daily.csv).

Outputs in runs/figures/:
  boris_error.png/.pdf       Aug–Oct 2024: precipitation, pressure, per-station daily error percentile
  rain_epoch.png/.pdf        network error percentile around the heaviest 3-day rain events + lag correlation
  rain_events_error.csv           per heavy-rain event: rain sum, network error percentile before / during / after
  boris_station_error.csv         per station: Boris-week error percentile and rank of that week in the record
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = os.path.join(ROOT, "figures")
STATIONS = ["prusacka", "za_prahem", "loupeznicka", "blativa", "kaple", "peklo", "jezero"]
NAME = {"prusacka": "Prušácká", "za_prahem": "Za prahem", "loupeznicka": "Loupežnická", "blativa": "Blátivá", "kaple": "Kaple", "peklo": "Peklo", "jezero": "Jezero"}
METHOD = "reconstruct_radon"
BLUE, ORANGE, AQUA, GRID, INK, INK2, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#e6e5e1", "#0b0b0b", "#52514e", "#8a8985"
rng = np.random.default_rng(0)

met = pd.read_csv("data/meteo/openmeteo_bozkov_daily.csv", parse_dates=["time"]).set_index("time")
met.index = met.index.tz_localize("UTC")
rain = met["precipitation_sum"]
rain3 = rain.rolling(3).sum()
pres = met["surface_pressure_min"]

# ---- daily error percentile per station -----------------------------------------------------
err = {}
for st in STATIONS:
    df = pd.read_csv(f"{ROOT}/main_suite/by_station/{st}/{METHOD}/sequence_scores.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    d = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()["relative_error"].resample("1D").mean()
    err[st] = d.rank(pct=True)  # percentile within the station's own record
err = pd.DataFrame(err)
network = err.mean(axis=1)  # mean percentile over stations with data
n_cov = err.notna().sum(axis=1)
network = network.where(n_cov >= 3)

# ---- Boris close-up --------------------------------------------------------------------------
w0, w1 = pd.Timestamp("2024-08-15", tz="UTC"), pd.Timestamp("2024-10-15", tz="UTC")
boris_week = slice(pd.Timestamp("2024-09-12", tz="UTC"), pd.Timestamp("2024-09-18", tz="UTC"))
rows = []
weekly = err.resample("7D").mean()
for st in STATIONS:
    s = err[st]
    bw = s.loc[boris_week].mean()
    wk = weekly[st].dropna()
    boris_wk = s.loc[boris_week].mean()
    rank = int((wk > boris_wk).sum()) + 1
    rows.append(dict(station=st, boris_week_mean_percentile=round(bw, 3), rank_of_boris_week=rank, n_weeks=len(wk),
                     pre_week_percentile=round(s.loc["2024-09-05":"2024-09-11"].mean(), 3),
                     post_week_percentile=round(s.loc["2024-09-19":"2024-09-25"].mean(), 3)))
boris_tab = pd.DataFrame(rows)
boris_tab.to_csv(os.path.join(OUT, "boris_station_error.csv"), index=False)
print(boris_tab.to_string(index=False))

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 2.6]})
ax = axes[0]
r = rain.loc[w0:w1]
ax.bar(r.index, r.values, width=1.0, color=BLUE)
ax.set_ylabel("precipitation\n(mm/day)")
ax.set_title("Storm Boris, 12–16 Sep 2024: 128 mm in 4 days (ERA5-Land, Bozkov)", loc="left", fontweight="bold")
ax.grid(axis="y")
ax = axes[1]
pp = pres.loc[w0:w1]
ax.plot(pp.index, pp.values, color=INK2, lw=1.2)
ax.set_ylabel("daily min surface\npressure (hPa)")
ax.grid(axis="y")
ax = axes[2]
offsets = np.arange(len(STATIONS))[::-1]
for st, off in zip(STATIONS, offsets):
    s = err[st].loc[w0:w1]
    ax.fill_between(s.index, off, off + s.values * 0.9, color=BLUE, alpha=0.75, step="mid")
    ax.axhline(off, color=GRID, lw=0.6)
# station names as y tick labels, i.e. outside the frame (inside text was crossed by the left spine)
ax.set_yticks(offsets + 0.45)
ax.set_yticklabels([NAME[st] for st in STATIONS], fontsize=10, color=INK)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.1, len(STATIONS) + 0.1)
ax.set_ylabel("")
ax.set_title("LSTM reconstruction: daily error as percentile of each station's record (0–1 per row)", loc="left", fontweight="bold")
for d in ["2024-09-12", "2024-09-16"]:
    for a in axes:
        a.axvline(pd.Timestamp(d, tz="UTC"), color=ORANGE, lw=1, ls="--")
axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"boris_error.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote boris_error")

# ---- heavy-rain events: epoch of network error percentile -----------------------------------
cover = network.notna()
r3 = rain3.reindex(network.index)
r3c = r3[cover]
thr = r3c.quantile(0.97)
cand = r3c[r3c >= thr].sort_values(ascending=False)
events = []
for d, v in cand.items():
    if all(abs((d - e).days) > 7 for e in events):
        events.append(d)
events = sorted(events)
print(f"\n3-day rain threshold (97th pct): {thr:.1f} mm, {len(events)} events")

LAGS = np.arange(-10, 15)


def epoch(series: pd.Series, dates):
    return np.array([series.reindex([d + pd.Timedelta(days=int(l)) for d in dates]).mean() for l in LAGS])


curve = epoch(network, events)
pool = network.index[cover]
null = np.array([epoch(network, rng.choice(pool, size=len(events), replace=False)) for _ in range(500)])
lo, hi = np.nanpercentile(null, 2.5, axis=0), np.nanpercentile(null, 97.5, axis=0)

rows = []
for d in events:
    rows.append(dict(
        end_of_3day_window=d.date(), rain_3day_mm=round(r3[d], 1),
        err_pct_before_m7_m3=round(network.loc[d - pd.Timedelta(days=7): d - pd.Timedelta(days=3)].mean(), 3),
        err_pct_during_m2_0=round(network.loc[d - pd.Timedelta(days=2): d].mean(), 3),
        err_pct_after_p1_p7=round(network.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=7)].mean(), 3),
        stations=int(n_cov.get(d, 0)),
    ))
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(OUT, "rain_events_error.csv"), index=False)
print(ev.to_string(index=False))

# lag correlation rain -> network error (rain leads for positive lag)
lags = np.arange(-7, 15)
xc = []
for l in lags:
    j = pd.concat([rain.shift(l).reindex(network.index), network], axis=1, keys=["rain", "err"]).dropna()
    xc.append(j.rain.corr(j.err, method="spearman"))
xc = np.array(xc)
# null band for the lag correlation via block-shuffled years is overkill; use random circular shifts of rain
null_xc = []
for _ in range(300):
    sh = int(rng.integers(30, len(rain) - 30))
    rs = pd.Series(np.roll(rain.values, sh), index=rain.index)
    j = pd.concat([rs.reindex(network.index), network], axis=1, keys=["rain", "err"]).dropna()
    null_xc.append(j.rain.corr(j.err, method="spearman"))
xlo, xhi = np.percentile(null_xc, [2.5, 97.5])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
ax = axes[0]
ax.fill_between(LAGS, lo, hi, color=GRID, label="95 % range of random dates")
ax.axhline(np.nanmean(network), color=MUTED, lw=1, ls="--", label="base")
ax.plot(LAGS, curve, color=BLUE, lw=2, marker="o", ms=4, label=f"heaviest 3-day rain events (n = {len(events)})")
boris_idx = [d for d in events if abs((d - pd.Timestamp("2024-09-15", tz="UTC")).days) <= 2]
if boris_idx:
    ax.plot(LAGS, epoch(network, boris_idx), color=ORANGE, lw=1.5, marker="s", ms=3, label="Storm Boris only")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("days relative to end of the 3-day rain window")
ax.set_ylabel("network mean error percentile")
ax.set_title("Reconstruction error around heavy rain", loc="left", fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y")
ax = axes[1]
ax.fill_between(lags, xlo, xhi, color=GRID, label="95 % range, shifted rain")
ax.plot(lags, xc, color=BLUE, lw=2, marker="o", ms=4)
ax.axhline(0, color=INK2, lw=0.8)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("lag (days): positive = rain precedes error")
ax.set_ylabel("Spearman corr(rain, network error percentile)")
ax.set_title("Daily rain vs daily error, all days", loc="left", fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"rain_epoch.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote rain_epoch")
print("\nlag corr:", dict(zip(lags.tolist(), np.round(xc, 3).tolist())), "null band", round(xlo, 3), round(xhi, 3))
