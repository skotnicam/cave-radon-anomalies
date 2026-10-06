#!/usr/bin/env python3
"""Leave-event-out control on the RADIM record (experiments/08_radim_leaveout.sh): around each qualifying event, share of
flagged hours and days and maximum relative error in +-10 days, versus the rest of the same +-30-day window
(days 11-30 before and after) as local reference, for the three seeds; the canonical in-sample model for comparison.
Output: <root>/figures/seismic_radim2017_leaveout.csv (printed).
"""
import sys
import numpy as np, pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else "runs"
EV = {"LGOM M4.8 2017-12-26": "2017-12-26 11:15", "LGOM M4.8 2019-01-29": "2019-01-29 12:53", "LGOM M4.9 2020-07-08": "2020-07-08 05:18",
      "Petrinja Mw6.4 2020-12-29": "2020-12-29 11:19", "Kahramanmaras Mw7.8 2023-02-06": "2023-02-06 01:17"}


def load(p):
    d = pd.read_csv(f"{p}/sequence_scores.csv"); d["timestamp"] = pd.to_datetime(d.timestamp, utc=True).dt.tz_localize(None)
    return d.set_index("timestamp").sort_index()


runs = {f"leave-out s{s}": load(f"{R}/radim_leaveout/s{s}") for s in (42, 1, 2)}
runs["in-sample (canonical)"] = load(f"{R}/radim_suite/by_station/bozkov2017/reconstruct_radon")
rows = []
for name, t in EV.items():
    t = pd.Timestamp(t)
    for rn, d in runs.items():
        dt = (d.index - t) / pd.Timedelta(days=1)
        near, far = d[np.abs(dt) <= 10], d[(np.abs(dt) > 10) & (np.abs(dt) <= 30)]
        rows.append(dict(event=name, model=rn, hours_flagged_near=near.is_anomaly.mean(), hours_flagged_far=far.is_anomaly.mean(),
                         days_flagged_near=f"{int(near.is_anomaly.resample('1D').max().sum())}/{near.is_anomaly.resample('1D').max().count()}",
                         max_err_near=near.relative_error.max(), max_err_far=far.relative_error.max(),
                         err_at_event_day=d.loc[t.normalize():t.normalize() + pd.Timedelta(days=1)].relative_error.max()))
T = pd.DataFrame(rows)
T.to_csv(f"{R}/figures/seismic_radim2017_leaveout.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 200)
print(T.round(3).to_string(index=False))
lo = T[T.model.str.startswith("leave-out")]
print(f"\nleave-out models, all events: flagged hours near {lo.hours_flagged_near.mean():.3f} vs far {lo.hours_flagged_far.mean():.3f}; "
      f"training-region nominal 0.05")
