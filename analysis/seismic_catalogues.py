#!/usr/bin/env python3
"""Catalogue-based seismicity tests against the radon flags of the final run (runs).

Response (Bozkov): daily fraction of stations flagged by the majority of the five reconstruction models
(figures/station_consensus_daily.csv), on days with at least 4 stations operating.
Response (Ochtinska): daily flag of the locally trained LSTM reconstruction.
Each event set is tested by the mean response in three windows relative to the event day,
before = days -7..-1, during = days 0..+1, after = days +2..+7, against 10 000 random sets of the same number
of dates drawn from the days with data (one-sided p = share of random sets with a mean >= the observed one),
and against 2 500 season-matched sets (one date per event day with a day of year within +-20 days, any year,
at least 30 days from the event), because the event sets cluster in particular seasons, and against every circular
shift of the whole event set by >= 30 days (primary test: it keeps the clustering of event days into episodes,
which the independent draws ignore; flags are autocorrelated over days). p values are two-sided unless noted.
Event sets
  swarm      West Bohemia swarm days: >= 20 tectonic events (Type T) per day in 49.9-50.6 N, 12.0-12.8 E
             (Czech bulletin, data/catalogues/czech_bulletin_2016_2026.csv)
  lgom       mining tremors M >= 4 in the Legnica-Glogow copper district, 51.2-51.8 N, 15.7-16.5 E
  dobro15    15 days of largest Dobrovolsky-weighted tectonic influence at Bozkov,
             influence = sum over events of 10**(0.43 M) / distance (EMSC + Czech bulletin type T, M >= 1)
  ochtinska  25 days of largest influence at Ochtinska from the ISC Slovakia catalogue, with and without the
             Zdana main-shock day (2023-10-09)
Output: <root>/figures/seismic_catalogue_tests.csv, printed summary.
Usage: python analysis/seismic_catalogues.py [runs_root]
"""
import sys
import numpy as np, pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "runs"
RNG = np.random.default_rng(7)
N_RANDOM = 10_000
SITES = {"bozkov": (50.65, 15.33), "ochtinska": (48.66, 20.30)}
WIN = {"before": (-7, -1), "during": (0, 1), "after": (2, 7)}


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---- responses -----------------------------------------------------------------------------
c = pd.read_csv(f"{R}/figures/station_consensus_daily.csv")
c["date"] = pd.to_datetime(c.date, utc=True).dt.tz_localize(None).dt.normalize()
c = c.groupby("date").sum(numeric_only=True)
resp_bozkov = (c.n_stations_flagged / c.n_stations_with_data).where(c.n_stations_with_data >= 4).dropna()
o = pd.read_csv(f"{R}/ochtinska_suite/by_cave/ochtinska/reconstruct_radon/sequence_scores.csv")
o["timestamp"] = pd.to_datetime(o.timestamp, utc=True).dt.tz_localize(None)
resp_ocht = o.set_index("timestamp").is_anomaly.resample("1D").max().dropna()

# ---- catalogues ----------------------------------------------------------------------------
cz = pd.read_csv("data/catalogues/czech_bulletin_2016_2026.csv", low_memory=False)
cz["day"] = pd.to_datetime(cz.Datetime).dt.normalize()
emsc = pd.read_csv("data/catalogues/emsc_catalog_2016_2026.csv")
emsc["day"] = pd.to_datetime(emsc.time, utc=True).dt.tz_localize(None).dt.normalize()
isc = pd.read_csv("data/catalogues/isc_slovakia_2019_2024.csv")
isc["day"] = pd.to_datetime(isc.time, utc=True).dt.tz_localize(None).dt.normalize()


def in_box(d, la0, la1, lo0, lo1, la="Lat", lo="Lon"):
    return d[(d[la] >= la0) & (d[la] <= la1) & (d[lo] >= lo0) & (d[lo] <= lo1)]


span_b = (resp_bozkov.index.min(), resp_bozkov.index.max())
tect = cz[cz.Type == "T"]
wb = in_box(tect, 49.9, 50.6, 12.0, 12.8).groupby("day").size()
swarm_days = sorted(d for d, n in wb.items() if n >= 20 and span_b[0] <= d <= span_b[1])
lg = in_box(cz[(cz.Type == "M") & (cz.mag >= 4)], 51.2, 51.8, 15.7, 16.5)
lgom_days = sorted({d for d in lg.day if span_b[0] <= d <= span_b[1]})

tb = pd.concat([emsc.rename(columns={"lat": "Lat", "lon": "Lon"})[["day", "Lat", "Lon", "mag"]],
                tect[["day", "Lat", "Lon", "mag"]]]).dropna()
tb = tb[tb.mag >= 1]
tb["infl"] = 10 ** (0.43 * tb.mag) / np.maximum(hav(*SITES["bozkov"], tb.Lat, tb.Lon), 1)
infl_b = tb.groupby("day").infl.sum()
infl_b = infl_b[(infl_b.index >= span_b[0]) & (infl_b.index <= span_b[1])]
dobro_days = sorted(infl_b.sort_values(ascending=False).head(15).index)

isc["infl"] = 10 ** (0.43 * isc.mag) / np.maximum(hav(*SITES["ochtinska"], isc.lat, isc.lon), 1)
infl_o = isc.groupby("day").infl.sum()
infl_o = infl_o[(infl_o.index >= resp_ocht.index.min()) & (infl_o.index <= resp_ocht.index.max())]
ocht_days = sorted(infl_o.sort_values(ascending=False).head(25).index)
ocht_wo = [d for d in ocht_days if d != pd.Timestamp("2023-10-09")]


def window_mean(resp, days, lo, hi):
    vals = [resp.get(d + pd.Timedelta(days=k)) for d in days for k in range(lo, hi + 1)]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return np.mean(vals) if vals else np.nan


def seasonal_draw(pool, days):
    """One random date per event day with a day of year within +-20 days (any year), excluding +-30 days of the event."""
    doy = pool.dayofyear.values
    out = []
    for d in days:
        dd = np.abs(((doy - d.dayofyear) + 182) % 365 - 182)
        cand = pool[(dd <= 20) & (np.abs((pool - d).days) > 30)]
        out.append(cand[RNG.integers(len(cand))] if len(cand) else pool[RNG.integers(len(pool))])
    return out


def shift_null(resp, days, lo, hi):
    """Circular shift of the whole event set by every offset >= 30 days (keeps the clustering of the events)."""
    idx = resp.index; t0, n = idx.min(), (idx.max() - idx.min()).days + 1
    vals = []
    for off in range(30, n - 30):
        sh = [t0 + pd.Timedelta(days=((d - t0).days + off) % n) for d in days]
        vals.append(window_mean(resp, sh, lo, hi))
    return np.array(vals)


def p2(r, obs):
    return min(1.0, 2 * min(np.mean(r >= obs), np.mean(r <= obs)))


def test(resp, days, name):
    pool = resp.index[(resp.index >= resp.index.min() + pd.Timedelta(days=7)) & (resp.index <= resp.index.max() - pd.Timedelta(days=7))]
    out = []
    for w, (lo, hi) in WIN.items():
        obs = window_mean(resp, days, lo, hi)
        rnd = np.array([window_mean(resp, list(RNG.choice(pool, size=len(days), replace=False)), lo, hi) for _ in range(N_RANDOM)])
        rs = np.array([window_mean(resp, seasonal_draw(pool, days), lo, hi) for _ in range(N_RANDOM // 4)])
        sh = shift_null(resp, days, lo, hi)
        out.append(dict(test=name, n=len(days), window=w, observed=obs, base_rate=resp.mean(),
                        random_mean=np.nanmean(rnd), p_random_two_sided=p2(rnd, obs),
                        seasonal_mean=np.nanmean(rs), p_seasonal_two_sided=p2(rs, obs),
                        shift_mean=np.nanmean(sh), p_shift_one_sided=np.mean(sh >= obs), p_shift_two_sided=p2(sh, obs)))
    return out


rows = []
rows += test(resp_bozkov, swarm_days, "West Bohemia swarm days")
rows += test(resp_bozkov, lgom_days, "LGOM mining tremors M>=4")
rows += test(resp_bozkov, dobro_days, "top-15 Dobrovolsky days, Bozkov")
rows += test(resp_ocht, ocht_days, "top-25 influence days, Ochtinska")
rows += test(resp_ocht, ocht_wo, "top-25 without Zdana day, Ochtinska")
T = pd.DataFrame(rows)
T.to_csv(f"{R}/figures/seismic_catalogue_tests.csv", index=False, float_format="%.4f")
pd.set_option("display.width", 160)
print(T.round(3).to_string(index=False))
print("\nswarm days:", [d.date() for d in swarm_days])
print("LGOM days:", [d.date() for d in lgom_days])
print("top-15 Dobrovolsky days (Bozkov):", [d.date() for d in dobro_days])
print("top-25 Ochtinska days include Zdana:", pd.Timestamp("2023-10-09") in ocht_days)
