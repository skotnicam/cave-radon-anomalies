#!/usr/bin/env python3
"""Microseisms at OKC (visually checked night windows, 2016-2026) vs. the radon anomaly flags.

Input: data/microseisms/daily_values.csv (one hour per day, 23:00-24:00 CET, moved by up to +-3 h when the window
contains a transient; status nominal / shifted / fallback / excluded / no_data; suspect flag).
Used: every day with a value, status nominal, shifted or fallback (main; the suspect days were verified by hand);
sensitivity ("all" rows in the CSV): nominal and shifted windows without the suspect days.
Intensity: rms_total_m_s (main) and rms_primary_m_s (primary microseisms), log10, deseasonalized by subtracting a
centred 31-day running median (>= 15 days present).
Responses
  stations  network error: daily mean LSTM relative error per station as a percentile of the station's record,
            averaged over stations with data (>= 4), deseasonalized; flags: daily fraction of flagged stations
            (station_consensus_daily.csv), 2022-2026
  radim     long reference-monitor record (2017-2026): daily max relative error of the LSTM model as a percentile,
            deseasonalized; flags: daily flag
Tests
  (b) Spearman correlation at lags -10..+10 d (positive lag = radon after microseisms) against 1 000 circular shifts
      of the microseism series (global p: max |rho| over lags vs the same for the shifts)
  (c) superposed epoch around the 20 strongest deseasonalized storms (>= 10 d apart, response present) against
      1 000 random sets of 20 dates (95 % band) and against circular shifts of the storm set (p of the window means
      before -7..-1, during 0..+1, after +2..+7)
Output: <root>/figures/microseisms.{png,pdf}, microseisms_v2_tests.csv, microseisms_v2_storms.csv; summary printed.
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = os.path.join(ROOT, "figures")
STATIONS = ["prusacka", "za_prahem", "kaple", "blativa", "loupeznicka", "peklo", "jezero"]
BLUE, ORANGE, AQUA, VIOLET, GREY, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#c9c9c9", "#52514e"
LAGS = np.arange(-10, 11)
WIN = {"before": (-7, -1), "during": (0, 1), "after": (2, 7)}
RNG = np.random.default_rng(5)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.titleweight": "bold"})


def deseason(s):
    return s - s.rolling(31, center=True, min_periods=15).median()


# ---- microseisms ------------------------------------------------------------------------------
M = pd.read_csv("data/microseisms/daily_values.csv", parse_dates=["date"]).set_index("date")
days = pd.date_range(M.index.min(), M.index.max(), freq="D")
M = M.reindex(days)
# all days with a value are used (the suspect days were verified by hand); sensitivity: without suspect days
valid_main = M.status.isin(["nominal", "shifted", "fallback"])
valid_all = M.status.isin(["nominal", "shifted"]) & (M.suspect != "yes")
coverage = dict(days=len(M), main=int(valid_main.sum()), without_suspect=int(valid_all.sum()),
                nominal=int((M.status == "nominal").sum()), shifted=int((M.status == "shifted").sum()))
print("coverage:", coverage)
ser = {}
for name, col in [("total", "rms_total_m_s"), ("primary", "rms_primary_m_s")]:
    for v, mask in [("main", valid_main), ("all", valid_all)]:
        ser[(name, v)] = deseason(np.log10(M[col].where(mask)))

# ---- responses --------------------------------------------------------------------------------
pct = {}
for st in STATIONS:
    d = pd.read_csv(f"{ROOT}/main_suite/by_station/{st}/reconstruct_radon/sequence_scores.csv")
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True).dt.tz_localize(None)
    pct[st] = d.set_index("timestamp").relative_error.resample("1D").mean().rank(pct=True)
P = pd.DataFrame(pct)
err_st = P.mean(axis=1).where(P.notna().sum(axis=1) >= 4)
c = pd.read_csv(f"{ROOT}/figures/station_consensus_daily.csv")
c["date"] = pd.to_datetime(c.date, utc=True).dt.tz_localize(None).dt.normalize()
c = c.groupby("date").sum(numeric_only=True)
flag_st = (c.n_stations_flagged / c.n_stations_with_data).where(c.n_stations_with_data >= 4)
r = pd.read_csv(f"{ROOT}/radim_suite/by_station/bozkov2017/reconstruct_radon/sequence_scores.csv")
r["timestamp"] = pd.to_datetime(r.timestamp, utc=True).dt.tz_localize(None)
r = r.set_index("timestamp")
err_rad = r.relative_error.resample("1D").max().rank(pct=True)
flag_rad = r.is_anomaly.resample("1D").max()
RESP = {"stations": (deseason(err_st.reindex(days)), flag_st.reindex(days)),
        "radim": (deseason(err_rad.reindex(days)), flag_rad.reindex(days))}


def lagcorr(x, y):
    return np.array([x.corr(y.shift(-L), method="spearman") for L in LAGS])


def window_mean(f, ds, lo, hi):
    v = [f.get(d + pd.Timedelta(days=k)) for d in ds for k in range(lo, hi + 1)]
    v = [x for x in v if x is not None and not np.isnan(x)]
    return np.mean(v) if v else np.nan


def epoch(f, ds):
    return np.array([np.nanmean([f.get(d + pd.Timedelta(days=int(L)), np.nan) for d in ds]) for L in LAGS])


rows, storms_out, figdat = [], [], {}
for rn, (err, flag) in RESP.items():
    span = err.dropna().index
    for (iname, v), x in ser.items():
        xx = x[(x.index >= span.min()) & (x.index <= span.max())]
        rc = lagcorr(xx, err)
        null = []
        xv = xx.values
        for _ in range(1000):
            k = RNG.integers(60, len(xv) - 60)
            null.append(np.nanmax(np.abs(lagcorr(pd.Series(np.roll(xv, k), index=xx.index), err))))
        p_glob = float(np.mean(np.array(null) >= np.nanmax(np.abs(rc))))
        row = dict(response=rn, intensity=iname, selection=v, n_days=int(xx.notna().sum()),
                   max_abs_rho=float(np.nanmax(np.abs(rc))), lag_of_max=int(LAGS[np.nanargmax(np.abs(rc))]),
                   rho_lag0=float(rc[LAGS == 0][0]), p_global=p_glob)
        # storms
        f = flag.dropna()
        cand = xx[xx.index.isin(f.index)].dropna().sort_values(ascending=False)
        storms = []
        for t in cand.index:
            if all(abs((t - s).days) >= 10 for s in storms) and span.min() + pd.Timedelta(days=10) <= t <= span.max() - pd.Timedelta(days=10):
                storms.append(t)
            if len(storms) == 20:
                break
        storms = sorted(storms)
        t0, n = f.index.min(), (f.index.max() - f.index.min()).days + 1
        for w, (lo, hi) in WIN.items():
            obs = window_mean(f, storms, lo, hi)
            sh = np.array([window_mean(f, [t0 + pd.Timedelta(days=((s - t0).days + o) % n) for s in storms], lo, hi)
                           for o in range(30, n - 30, 3)])
            sh = sh[~np.isnan(sh)]
            row[f"epoch_{w}"] = obs
            row[f"p_shift_{w}"] = min(1.0, 2 * min(np.mean(sh >= obs), np.mean(sh <= obs)))
        row["base_rate"] = float(f.mean())
        rows.append(row)
        if v == "main":
            pool = f.index[(f.index >= span.min() + pd.Timedelta(days=10)) & (f.index <= span.max() - pd.Timedelta(days=10))]
            rnd = np.array([epoch(f, list(RNG.choice(pool, size=len(storms), replace=False))) for _ in range(1000)])
            figdat[(rn, iname)] = dict(rc=rc, null_hi=np.nanpercentile(null, 95), ep=epoch(f, storms),
                                      lo=np.nanpercentile(rnd, 2.5, axis=0), hi=np.nanpercentile(rnd, 97.5, axis=0),
                                      base=f.mean(), storms=storms)
            for s in storms:
                storms_out.append(dict(response=rn, intensity=iname, date=s.date(), rms_total_um_s=M.rms_total_m_s.get(s) * 1e6,
                                       status=M.status.get(s), flag=f.get(s)))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "microseisms_v2_tests.csv"), index=False, float_format="%.4f")
pd.DataFrame(storms_out).to_csv(os.path.join(OUT, "microseisms_v2_storms.csv"), index=False, float_format="%.3f")
pd.set_option("display.width", 220)
print(T.round(3).to_string(index=False))

# ---- figure -------------------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.62, wspace=0.26)
ax = fig.add_subplot(gs[0, :])
tot = M.rms_total_m_s.where(valid_main) * 1e6
ax.plot(tot.index, tot, color=GREY, lw=0.6, label="daily RMS, 0.05–0.3 Hz (checked night hour)")
ax.plot(tot.index, tot.rolling(31, center=True, min_periods=15).median(), color=BLUE, lw=1.8, label="31-day running median")
st_dates = figdat[("radim", "total")]["storms"]
ax.scatter(st_dates, [tot.get(s) for s in st_dates], s=20, color=ORANGE, zorder=3, label="20 strongest storms (deseasonalized)")
sh = M.index[(M.status == "shifted").values]
ax.scatter(sh, np.full(len(sh), np.nanmin(tot) * 0.7), marker="|", s=18, color=INK2, lw=0.6, label="window shifted after visual check")
ax.set_yscale("log"); ax.set_ylabel("velocity RMS (µm/s)")
ax.set_title("(a) Microseisms at OKC, vertical component, 23:00–24:00 CET", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

ax = fig.add_subplot(gs[1, 0])
for (rn, iname), col, lab in [(("stations", "total"), BLUE, "stations 2022–26, total"), (("radim", "total"), ORANGE, "RADIM 2017–26, total"),
                              (("radim", "primary"), VIOLET, "RADIM 2017–26, primary")]:
    dd = figdat[(rn, iname)]
    ax.plot(LAGS, dd["rc"], color=col, lw=1.8, marker="o", ms=3, label=lab)
hi = max(figdat[k]["null_hi"] for k in figdat)
ax.axhspan(-hi, hi, color=GREY, alpha=0.4, lw=0, label="95 % of max |ρ|, shifts")
ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6, ls=":")
ax.set_ylim(-0.2, 0.25); ax.set_xlabel("lag (days; > 0: radon after microseisms)"); ax.set_ylabel("Spearman ρ")
ax.set_title("(b) Correlation with the radon error", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

ax = fig.add_subplot(gs[1, 1])
for rn, col, lab in [("stations", BLUE, "stations, flagged fraction"), ("radim", ORANGE, "RADIM, flagged days")]:
    dd = figdat[(rn, "total")]
    ax.fill_between(LAGS, dd["lo"], dd["hi"], color=col, alpha=0.12, lw=0)
    ax.plot(LAGS, dd["ep"], color=col, lw=1.8, marker="o", ms=3, label=lab)
    ax.axhline(dd["base"], color=col, lw=0.8, ls="--")
ax.axvline(0, color=INK2, lw=0.6, ls=":")
ax.set_ylim(0, 0.4); ax.set_xlabel("days from storm"); ax.set_ylabel("flags")
ax.set_title("(c) Flags around the 20 strongest storms", loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"microseisms.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
print("wrote microseisms")
