#!/usr/bin/env python3
"""Do anomaly flags line up with deep low-pressure systems (e.g. Storm Boris, Sep 2024)?

Uses the barometer of the RADIM reference monitor (data/meteo/radim_jezero_pressure_hourly.csv), the Bozkov network flags
from runs (majority of the 5 reconstruction models per station-day) and the

Outputs into runs/figures/:
  pressure_low_events.csv        each deep low: date, min pressure, 48 h drop, flags before/at/after
  pressure_epoch_bozkov.csv      superposed-epoch flag fraction by lag (network)
  pressure_epoch.png/.pdf   epoch curves + Boris close-up
  boris_window_flags.csv         day-by-day flags of the stations around Storm Boris
"""
from __future__ import annotations

import glob
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
RECON = [
    "reconstruct_radon",
    "reconstruct_delayed_temp",
    "transformer_reconstruct_radon",
    "transformer_reconstruct_delayed_temp",
    "conv_reconstruct_radon",
]
BLUE, ORANGE, AQUA, GRID, INK2, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#e6e5e1", "#52514e", "#8a8985"
LAGS = np.arange(-10, 11)
rng = np.random.default_rng(0)


def daily_flags(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df["is_anomaly"].astype(int).resample("1D").max()


# ---- pressure ------------------------------------------------------------------------------
# The station files carry pressure copied from tlak.csv only until 2024-03-08 and a repeated
# previous-year copy afterwards. Use the reference monitor's barometer
# (same instrument as tlak.csv, corr 1.000) which runs to 2026-05.
p = pd.read_csv("data/meteo/radim_jezero_pressure_hourly.csv", parse_dates=["time_utc"]).set_index("time_utc")["pressure_hPa"]
hourly = p.resample("1h").mean()  # regular hourly index, gaps as NaN
daily_min = hourly.resample("1D").min()
daily_mean = hourly.resample("1D").mean()
drop48 = (hourly.rolling(48).max() - hourly).resample("1D").max()  # largest fall over the past 48 h

# ---- flags ---------------------------------------------------------------------------------
net = {}
for st in STATIONS:
    cols = []
    for m in RECON:
        f = f"{ROOT}/main_suite/by_station/{st}/{m}/sequence_scores.csv"
        if os.path.exists(f):
            cols.append(daily_flags(f).rename(m))
    d = pd.concat(cols, axis=1)
    net[st] = ((d.sum(axis=1) >= 3) & d.notna().any(axis=1)).astype(float).where(d.notna().any(axis=1))
net = pd.DataFrame(net)
caves = pd.DataFrame(index=net.index)  # no single-monitor caves in the published analysis

# ---- deep-low events -----------------------------------------------------------------------
cover = net.notna().any(axis=1)
period = daily_min[cover.reindex(daily_min.index, fill_value=False)]
thr = period.quantile(0.03)
cand = period[period <= thr].sort_values()
events = []
for d, v in cand.items():
    if all(abs((d - e).days) > 5 for e in events):
        events.append(d)
events = sorted(events)
print(f"pressure threshold (3rd pct of daily min over network period): {thr:.1f} hPa, {len(events)} events")


def window_stats(flags: pd.DataFrame, d: pd.Timestamp, a: int, b: int):
    w = flags.loc[d + pd.Timedelta(days=a): d + pd.Timedelta(days=b)]
    if w.notna().sum().sum() == 0:
        return np.nan, 0
    per_station = w.max(axis=0)  # station flagged at least once in the window
    return per_station.sum(), per_station.notna().sum()


rows = []
for d in events:
    fb, nb = window_stats(net, d, -5, -1)
    fa, na = window_stats(net, d, 0, 3)
    rows.append(
        dict(
            date=d.date(),
            min_pressure_hPa=round(daily_min[d], 1),
            drop48h_hPa=round(drop48.get(d, np.nan), 1),
            stations_with_data=int(na),
            stations_flagged_before_m5_m1=fb,
            stations_flagged_at_0_p3=fa,
        )
    )
ev = pd.DataFrame(rows)
ev.to_csv(os.path.join(OUT, "pressure_low_events.csv"), index=False)
print(ev.to_string(index=False))


# ---- superposed epoch ----------------------------------------------------------------------
def epoch_curve(flags: pd.DataFrame, dates) -> pd.Series:
    vals = []
    for lag in LAGS:
        idx = [d + pd.Timedelta(days=int(lag)) for d in dates]
        w = flags.reindex(idx)
        vals.append(w.stack().mean() if w.notna().sum().sum() else np.nan)
    return pd.Series(vals, index=LAGS)


base = net.stack().mean()
curve = epoch_curve(net, events)
# null: same number of random dates drawn from days with network coverage
pool = net.index[cover]
null = np.array([epoch_curve(net, rng.choice(pool, size=len(events), replace=False)).values for _ in range(500)])
lo, hi = np.nanpercentile(null, 2.5, axis=0), np.nanpercentile(null, 97.5, axis=0)
ep = pd.DataFrame({"lag_days": LAGS, "flag_fraction": curve.values, "null_2.5": lo, "null_97.5": hi, "base": base})
ep.to_csv(os.path.join(OUT, "pressure_epoch_bozkov.csv"), index=False, float_format="%.4f")
print("\nbase network flag fraction %.3f" % base)
print(ep.round(3).to_string(index=False))

# ---- Boris close-up ------------------------------------------------------------------------
b0, b1 = pd.Timestamp("2024-09-05", tz="UTC"), pd.Timestamp("2024-09-25", tz="UTC")
win = pd.concat([daily_min.rename("pressure_min"), net.add_prefix("bozkov_"), caves.add_prefix("cave_")], axis=1).loc[b0:b1]
win.index = win.index.date
win.to_csv(os.path.join(OUT, "boris_window_flags.csv"), float_format="%.0f")
print("\nStorm Boris window:")
print(win.to_string())

# ---- figure --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
ax.fill_between(LAGS, lo, hi, color=GRID, label="95 % range of random dates")
ax.axhline(base, color=MUTED, lw=1, ls="--", label=f"base rate {base:.3f}")
ax.plot(LAGS, curve.values, color=BLUE, lw=2, marker="o", ms=4, label=f"deep lows (n = {len(events)})")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("days relative to pressure minimum")
ax.set_ylabel("fraction of station-days flagged")
ax.set_title("Network flags around 18 deep lows", loc="left", fontweight="bold")
ax.set_ylim(top=max(np.nanmax(hi), np.nanmax(curve.values)) * 1.35)  # room for the legend above the data
ax.legend(loc="upper left", fontsize=9.5, ncol=1, framealpha=0.9)
ax.grid(axis="y")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
pr = hourly.loc[b0:b1]
ax.plot(pr.index, pr.values, color=INK2, lw=1)
ax.set_ylabel("pressure (hPa)")
ax.set_title("Storm Boris, Sep 2024: pressure and daily flags", loc="left", fontweight="bold")
ax2 = ax.twinx()  # second axis only carries categorical flag rows, no second scale
# reference monitor (RADIM) at the Jezero site, long record used in the paper; the other caves end before 2024-09
_rp = f"{ROOT}/radim_suite/by_station/bozkov2017/reconstruct_radon/sequence_scores.csv"
ref = pd.DataFrame({"radim": daily_flags(_rp).astype(float)})
NAME["radim"] = "Jezero (RADIM)"
rows_ = list(net.columns) + ["radim"]
for i, name in enumerate(rows_):
    s = (net if name in net.columns else ref)[name].loc[b0:b1]
    col = BLUE if name in net.columns else AQUA
    on = s[s == 1]
    ax2.scatter(on.index + pd.Timedelta(hours=12), np.full(len(on), i), s=28, color=col, marker="s")
    off = s[s == 0]
    ax2.scatter(off.index + pd.Timedelta(hours=12), np.full(len(off), i), s=6, color=GRID, marker="s")
ax2.set_yticks(range(len(rows_)))
ax2.set_yticklabels([NAME.get(r, r) for r in rows_], fontsize=9)
ax2.set_ylim(-1, len(rows_))
ax2.spines[["top"]].set_visible(False)
ax.spines[["top"]].set_visible(False)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.text(0.01, 0.03, "square = flagged that day (blue: stations, green: reference monitor)", transform=ax.transAxes,
        fontsize=9, color=INK2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"pressure_epoch.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote pressure_epoch")
