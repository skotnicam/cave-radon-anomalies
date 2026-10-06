#!/usr/bin/env python3
"""Seismicity tests on the long reference-monitor record (RADIM at the Jezero site, May 2017 - April 2026).

Responses (daily flag of the LSTM radon reconstruction; no network consensus exists before 2022):
  head      canonical model, trained on the earliest 70 % (to Aug 2023), whole record scored
  tail_sNN  control models trained on the last 30 % (Aug 2023 - Apr 2026; (not part of the published repository)). NOT USED by
            default (INCLUDE_TAIL = False): they flag ~47 % of the days of 2017-2023 because the radon level of the
            early years differs from that of 2023-2026, so their out-of-sample flags reflect the change of regime and
            not events. The out-of-sample control is the leave-event-out model (radim_leaveout.py).
Event sets (restricted to the days of each response):
  qualifying   every event with distance / Dobrovolsky radius <= 1.05 (R = 10**(0.43 M) km), USGS global
               (M >= 5.5 2017-2022, M >= 6 2022-2026) and EMSC regional
  swarm        West Bohemia swarm days (>= 20 tectonic events per day, 49.9-50.6 N, 12.0-12.8 E, Czech bulletin)
  lgom         mining tremors M >= 4 in the Legnica-Glogow district (51.2-51.8 N, 15.7-16.5 E, Czech bulletin)
  dobro15      15 days of largest Dobrovolsky-weighted influence at Bozkov, influence = sum of 10**(0.43 M)/distance
               per catalogue and day; the daily maximum over EMSC, USGS and Czech tectonic events is used (no double count)
Test: mean response before (-7..-1), during (0..+1) and after (+2..+7) the event days against 10 000 random date sets
and against every circular shift of the whole event set by >= 30 days (primary; keeps the clustering of events).
Also: flags and maximum error within +-10 days of each qualifying event, for every model.
Output: <root>/figures/seismic_radim2017_tests.csv, seismic_radim2017_qualifying.csv; printed summary.
"""
import glob, os, sys
import numpy as np, pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "runs"
RNG = np.random.default_rng(11)
BOZ = (50.65, 15.33)
WIN = {"before": (-7, -1), "during": (0, 1), "after": (2, 7)}


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, [la1, lo1, la2, lo2])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def daily(path):
    d = pd.read_csv(f"{path}/sequence_scores.csv")
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True).dt.tz_localize(None)
    d = d.set_index("timestamp")
    return d.is_anomaly.resample("1D").max().dropna(), d.relative_error.resample("1D").max().dropna()


resp, errs = {}, {}
resp["head"], errs["head"] = daily(f"{R}/radim_suite/by_station/bozkov2017/reconstruct_radon")
INCLUDE_TAIL = False
for p in sorted(glob.glob(f"{R}/radim_tail_models/s*")) if INCLUDE_TAIL else []:
    if not os.path.exists(f"{p}/sequence_scores.csv"):
        continue
    f, e = daily(p)
    ts = pd.to_datetime(pd.read_csv(f"{p}/sequence_scores.csv", usecols=["timestamp"]).timestamp, utc=True).dt.tz_localize(None)
    n_train = max(1, int(len(ts) * 0.3))  # chronological_train_val_split, train_region="tail"
    cut = ts.iloc[len(ts) - n_train].normalize()  # first day of the tail training region
    name = f"tail_{os.path.basename(p)}"
    resp[name], errs[name] = f[f.index < cut], e[e.index < cut]
    print(f"{name}: out-of-sample part {resp[name].index.min().date()} .. {resp[name].index.max().date()}")

# ---- catalogues -------------------------------------------------------------------------------
cz = pd.read_csv("data/catalogues/czech_bulletin_2016_2026.csv", low_memory=False); cz["day"] = pd.to_datetime(cz.Datetime).dt.normalize()
em = pd.read_csv("data/catalogues/emsc_catalog_2016_2026.csv"); em["t"] = pd.to_datetime(em.time, utc=True).dt.tz_localize(None)
us = pd.concat([pd.read_csv(f) for f in ["data/catalogues/usgs_global_M55_2017_2022.csv", "data/catalogues/usgs_global_M6_2022_2026.csv"]])
us["t"] = pd.to_datetime(us.time, utc=True).dt.tz_localize(None)
cats = {"emsc": em.rename(columns={"lat": "la", "lon": "lo"})[["t", "la", "lo", "mag"]],
        "usgs": us.rename(columns={"latitude": "la", "longitude": "lo"})[["t", "la", "lo", "mag"]],
        "cz_tect": cz[cz.Type == "T"].rename(columns={"Datetime": "t", "Lat": "la", "Lon": "lo"})[["t", "la", "lo", "mag"]]}
for k, c in cats.items():
    c = c.dropna().copy(); c["t"] = pd.to_datetime(c.t); c["day"] = c.t.dt.normalize()
    c["R"] = 10 ** (0.43 * c.mag); c["d"] = hav(*BOZ, c.la, c.lo); c["ratio"] = c.d / c.R
    c["infl"] = c.R / np.maximum(c.d, 1)
    cats[k] = c
qual = pd.concat([cats["usgs"], cats["emsc"]]).query("ratio <= 1.05").sort_values("t")
qual = qual.drop_duplicates(subset="day")
infl = pd.concat([c[c.mag >= 1].groupby("day").infl.sum() for c in cats.values()], axis=1).max(axis=1)
tect = cz[cz.Type == "T"]
wb = tect[(tect.Lat.between(49.9, 50.6)) & (tect.Lon.between(12.0, 12.8))].groupby("day").size()
swarm_all = sorted(wb[wb >= 20].index)
lg = cz[(cz.Type == "M") & (cz.mag >= 4) & cz.Lat.between(51.2, 51.8) & cz.Lon.between(15.7, 16.5)]
lgom_all = sorted(set(lg.day))


def window_mean(r, days, lo, hi):
    v = [r.get(d + pd.Timedelta(days=k)) for d in days for k in range(lo, hi + 1)]
    v = [x for x in v if x is not None and not np.isnan(x)]
    return np.mean(v) if v else np.nan


def shift_null(r, days, lo, hi):
    t0, n = r.index.min(), (r.index.max() - r.index.min()).days + 1
    return np.array([window_mean(r, [t0 + pd.Timedelta(days=((d - t0).days + o) % n) for d in days], lo, hi)
                     for o in range(30, n - 30, 3)])


def p2(x, obs):
    x = x[~np.isnan(x)]
    return min(1.0, 2 * min(np.mean(x >= obs), np.mean(x <= obs))) if len(x) else np.nan


rows = []
for rn, r in resp.items():
    lo_d, hi_d = r.index.min() + pd.Timedelta(days=7), r.index.max() - pd.Timedelta(days=7)
    inside = lambda ds: [d for d in ds if lo_d <= d <= hi_d and d in r.index]
    top = infl[(infl.index >= lo_d) & (infl.index <= hi_d)].sort_values(ascending=False)
    sets = {"qualifying (d/R <= 1.05)": inside(sorted(qual.day)), "West Bohemia swarm days": inside(swarm_all),
            "LGOM tremors M>=4": inside(lgom_all), "top-15 Dobrovolsky days": inside(sorted(top.head(15).index))}
    pool = r.index[(r.index >= lo_d) & (r.index <= hi_d)]
    for sn, days in sets.items():
        if not days:
            continue
        for w, (a, b) in WIN.items():
            obs = window_mean(r, days, a, b)
            rnd = np.array([window_mean(r, list(RNG.choice(pool, size=len(days), replace=False)), a, b) for _ in range(4000)])
            sh = shift_null(r, days, a, b)
            rows.append(dict(response=rn, test=sn, n=len(days), window=w, observed=obs, base_rate=r.mean(),
                             p_random_two_sided=p2(rnd, obs), p_shift_one_sided=np.mean(sh[~np.isnan(sh)] >= obs),
                             p_shift_two_sided=p2(sh, obs)))
T = pd.DataFrame(rows)
T.to_csv(f"{R}/figures/seismic_radim2017_tests.csv", index=False, float_format="%.4f")

q = []
for _, e in qual.iterrows():
    row = dict(date=e.day.date(), mag=e.mag, R_km=round(e.R), dist_km=round(e.d), ratio=round(e.ratio, 2))
    for rn in resp:
        f, er = resp[rn], errs[rn]
        a, b = e.day - pd.Timedelta(days=10), e.day + pd.Timedelta(days=10)
        w = f.loc[a:b]
        row[f"{rn}_flagged_days"] = f"{int(w.sum())}/{len(w)}" if len(w) else "-"
        row[f"{rn}_max_err"] = round(er.loc[a:b].max(), 2) if len(er.loc[a:b]) else np.nan
    q.append(row)
Q = pd.DataFrame(q)
Q.to_csv(f"{R}/figures/seismic_radim2017_qualifying.csv", index=False)
pd.set_option("display.width", 220)
print("\nQualifying events (+-10 days):"); print(Q.to_string(index=False))
print("\nTests:"); print(T.round(3).to_string(index=False))
