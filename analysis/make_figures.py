#!/usr/bin/env python3
"""Summary figures of the main suite (ten methods, seven stations).

Outputs (PNG + PDF) into <root>/figures/ plus a few CSV tables:
  method_summary       per-method anomaly rate / drift / error-level correlation, station spread
  method_agreement     Jaccard agreement of flagged samples between methods, per station + median
  station_consensus    daily number of Bozkov stations flagged by the reconstruction family
  negative_examples    za_prahem: flags of LSTM reconstruction vs meteo_to_radon / MMLR / EMD-SVR
  transfer_penalty     cross-cave transfer matrices (anomaly rate, median relative error)
  bozkov_vs_jezero     long-term Bozkov monitor vs the jezero station (two instruments)
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- palette (validated reference palette, light mode) -------------------------------------
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"

METHODS = [
    "reconstruct_radon",
    "reconstruct_delayed_temp",
    "transformer_reconstruct_radon",
    "transformer_reconstruct_delayed_temp",
    "conv_reconstruct_radon",
    "forecast",
    "conv_forecast",
    "meteo_to_radon",
    "mmlr",
    "emd_svr",
]
RECON = METHODS[:5]
FAMILY = {m: "Reconstruction" for m in RECON}
FAMILY.update({"forecast": "Forecast", "conv_forecast": "Forecast"})
FAMILY.update({"meteo_to_radon": "Covariate regression", "mmlr": "Covariate regression"})
FAMILY["emd_svr"] = "Signal decomposition"
FAMILY_COLOR = {
    "Reconstruction": BLUE,
    "Forecast": AQUA,
    "Covariate regression": ORANGE,
    "Signal decomposition": VIOLET,
}
LABEL = {
    "reconstruct_radon": "LSTM reconstruct",
    "reconstruct_delayed_temp": "LSTM reconstruct, lagged T",
    "transformer_reconstruct_radon": "Transformer reconstruct",
    "transformer_reconstruct_delayed_temp": "Transformer reconstruct, lagged T",
    "conv_reconstruct_radon": "Conv1D reconstruct",
    "forecast": "LSTM forecast",
    "conv_forecast": "Conv1D forecast",
    "meteo_to_radon": "LSTM meteo→radon",
    "mmlr": "MMLR",
    "emd_svr": "EMD–SVR",
}
STATIONS = ["prusacka", "za_prahem", "loupeznicka", "blativa", "kaple", "peklo", "jezero"]
NAME = {"prusacka": "Prušácká", "za_prahem": "Za prahem", "loupeznicka": "Loupežnická", "blativa": "Blátivá", "kaple": "Kaple", "peklo": "Peklo", "jezero": "Jezero"}
MARK_DATES = {"2024-09-14": "Storm Boris", "2025-11-22": "2025-11-22 event"}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
    }
)


def read_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df["is_anomaly"] = df["is_anomaly"].astype(int)
    return df


def save(fig, out_dir: str, name: str):
    fig.savefig(os.path.join(out_dir, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---- load Bozkov suite --------------------------------------------------------------------
def load_suite(root: str) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for p in sorted(glob.glob(f"{root}/main_suite/by_station/*/*/sequence_scores.csv")):
        st, m = p.split(os.sep)[-3:-1]
        if st in STATIONS and m in METHODS:
            out[(st, m)] = read_scores(p)
    return out


def metrics_table(suite) -> pd.DataFrame:
    rows = []
    for (st, m), df in suite.items():
        f = df["is_anomaly"].values
        h = len(f) // 2
        radon_col = [c for c in df.columns if c.startswith("ARn")][0]
        corr = np.corrcoef(df["reconstruction_error"], df[radon_col])[0, 1]
        rows.append(
            dict(
                station=st,
                method=m,
                family=FAMILY[m],
                n=len(f),
                anomaly_rate=f.mean(),
                rate_h1=f[:h].mean(),
                rate_h2=f[h:].mean(),
                drift=abs(f[h:].mean() - f[:h].mean()),
                abs_corr_err_radon=abs(corr),
            )
        )
    return pd.DataFrame(rows)


def fig1_method_summary(met: pd.DataFrame, out_dir: str, ref: pd.DataFrame | None = None):
    """ref: optional metrics of the long RADIM reference record (drawn as hollow diamonds, not in the medians)."""
    cols = [
        ("anomaly_rate", "Anomaly rate"),
        ("drift", "Drift between\nrecord halves"),
        ("abs_corr_err_radon", "|corr(score,\nradon level)|"),
    ]
    cols[0] = ("anomaly_rate", "Anomaly rate\n ")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.9), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.8, bottom=0.2, wspace=0.12)
    y = np.arange(len(METHODS))[::-1]
    for ax, (col, title) in zip(axes, cols):
        for yi, m in zip(y, METHODS):
            sub = met[met.method == m]
            c = FAMILY_COLOR[FAMILY[m]]
            ax.scatter(sub[col], np.full(len(sub), yi), s=18, color=c, alpha=0.45, edgecolors="none")
            med = sub[col].median()
            ax.scatter([med], [yi], s=70, color=c, edgecolors="white", linewidths=1.2, zorder=3)
            ax.text(med, yi + 0.32, f"{med:.2f}", ha="center", va="bottom", fontsize=9, color=INK2)
            if ref is not None and m in set(ref.method):
                ax.scatter(ref.loc[ref.method == m, col], [yi], s=46, marker="D", facecolors="none",
                           edgecolors=INK, linewidths=1.1, zorder=4)
        ax.set_title(title, loc="left")
        ax.grid(axis="x")
        ax.set_xlim(left=0)
        ax.axvline(0, color=GRID)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[m] for m in METHODS])
    for tick, m in zip(axes[0].get_yticklabels(), METHODS):
        tick.set_color(INK)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=f)
        for f, c in FAMILY_COLOR.items()
    ]
    if ref is not None:
        handles.append(plt.Line2D([], [], marker="D", ls="", markerfacecolor="none", markeredgecolor=INK,
                                  markersize=6, label="RADIM at Jezero, 2017–2026"))
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5 if ref is not None else 4,
                   fontsize=9 if ref is not None else None)
    fig.suptitle(
        "Bozkov, 7 stations × 10 methods: large dot = median over stations, small = single station"
        + ("; diamond = long reference record (not in the median)" if ref is not None else ""),
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK2,
    )
    save(fig, out_dir, "method_summary")


def jaccard(a: pd.Series, b: pd.Series) -> float:
    j = pd.concat([a, b], axis=1, keys=["a", "b"], join="inner")
    inter = ((j.a == 1) & (j.b == 1)).sum()
    union = ((j.a == 1) | (j.b == 1)).sum()
    return inter / union if union else np.nan


def fig2_method_agreement(suite, out_dir: str):
    mats = {}
    for st in STATIONS:
        M = np.full((len(METHODS), len(METHODS)), np.nan)
        for i, a in enumerate(METHODS):
            for k, b in enumerate(METHODS):
                if (st, a) in suite and (st, b) in suite:
                    M[i, k] = jaccard(suite[(st, a)]["is_anomaly"], suite[(st, b)]["is_anomaly"])
        mats[st] = M
    median = np.nanmedian(np.stack(list(mats.values())), axis=0)
    panels = [(st, mats[st]) for st in STATIONS] + [("median over stations", median)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 9))
    fig.subplots_adjust(wspace=0.12, hspace=0.9, right=0.9)
    short = [LABEL[m] for m in METHODS]
    for idx, (ax, (title, M)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(NAME.get(title, title), loc="left")
        ax.set_xticks(range(len(METHODS)))
        ax.set_yticks(range(len(METHODS)))
        ax.set_xticklabels(short, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(short if idx % 4 == 0 else [], fontsize=8)
        for i in range(len(METHODS)):
            for k in range(len(METHODS)):
                v = M[i, k]
                if np.isfinite(v):
                    ax.text(k, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > 0.55 else INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Jaccard index of flagged samples")
    cb.outline.set_visible(False)
    fig.suptitle("Agreement of anomaly flags between methods", x=0.01, ha="left", color=INK2)
    save(fig, out_dir, "method_agreement")
    pd.DataFrame(median, index=METHODS, columns=METHODS).to_csv(
        os.path.join(out_dir, "method_agreement_jaccard_median.csv"), float_format="%.3f"
    )


def station_days(suite):
    """Daily table: for each station, number of reconstruction methods flagging any sample that day."""
    per_station = {}
    for st in STATIONS:
        cols = []
        for m in RECON:
            if (st, m) in suite:
                cols.append(suite[(st, m)]["is_anomaly"].resample("1D").max().rename(m))
        if cols:
            d = pd.concat(cols, axis=1)
            per_station[st] = d
    return per_station


def fig3_station_consensus(suite, out_dir: str):
    per = station_days(suite)
    idx = sorted(set().union(*[d.index for d in per.values()]))
    cover = pd.DataFrame(index=idx)
    flagged = pd.DataFrame(index=idx)
    for st, d in per.items():
        has = d.notna().any(axis=1)
        cover[st] = has.reindex(idx, fill_value=False).astype(bool)
        flagged[st] = ((d.sum(axis=1) >= 3) & has).reindex(idx, fill_value=False).astype(bool)  # majority of 5 recon methods
    n_cov = cover.sum(axis=1)
    n_flag = flagged.sum(axis=1)
    frac = (n_flag / n_cov.where(n_cov >= 3)).fillna(0)  # fraction only meaningful with >= 3 stations
    out = pd.DataFrame({"n_stations_with_data": n_cov, "n_stations_flagged": n_flag, "fraction": frac})
    out.index.name = "date"
    out.to_csv(os.path.join(out_dir, "station_consensus_daily.csv"), float_format="%.3f")

    # one panel: the flagged fraction is only the ratio of the two quantities shown here (still written to the CSV)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(n_cov.index, 0, n_cov.values, step="mid", color=GRID, label="stations with data")
    ax.bar(n_flag.index, n_flag.values, width=1.0, color=BLUE, label="stations flagged (majority of 5 reconstruction models)")
    ax.set_ylabel("number of stations")
    ax.set_ylim(0, 8.4)
    ax.set_yticks(range(0, 8))
    ax.set_title("Bozkov network: how many stations flag the same day", loc="left")
    ax.grid(axis="y")
    for d, name in MARK_DATES.items():
        t = pd.Timestamp(d, tz="UTC")
        ax.axvline(t, color=ORANGE, lw=1, ls="--")
        ax.text(t, 8.3, name, color=ORANGE, fontsize=10, ha="center", va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, frameon=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    save(fig, out_dir, "station_consensus")

    top = out[out.n_stations_with_data >= 4].sort_values(["n_stations_flagged", "fraction"], ascending=False).head(25)
    top.to_csv(os.path.join(out_dir, "station_consensus_top_days.csv"), float_format="%.3f")


def fig4_negative_examples(suite, out_dir: str, station: str = "za_prahem"):
    show = ["reconstruct_radon", "meteo_to_radon", "mmlr", "emd_svr"]
    fig, axes = plt.subplots(len(show), 1, figsize=(9, 9), sharex=True)
    for ax, m in zip(axes, show):
        df = suite[(station, m)]
        radon_col = [c for c in df.columns if c.startswith("ARn")][0]
        r = df[radon_col]
        ax.plot(r.index, r.values, color=MUTED, lw=0.6)
        fl = df[df.is_anomaly == 1]
        c = FAMILY_COLOR[FAMILY[m]]
        ax.scatter(fl.index, fl[radon_col], s=6, color=c, edgecolors="none", zorder=3)
        rate = df.is_anomaly.mean()
        corr = np.corrcoef(df["reconstruction_error"], r)[0, 1]
        note = ", fitted on the whole series" if m == "emd_svr" else ""
        ax.set_title(f"{LABEL[m]}: {rate*100:.1f}% flagged, corr(score, radon) = {corr:+.2f}{note}", loc="left", fontsize=11)
        # held-out region: everything after the first 70 % of the scored sequences (the training region);
        # EMD-SVR decomposes and fits the whole series, so it has no held-out part
        k = int(len(df) * 0.7)
        if m != "emd_svr" and 0 < k < len(df):
            t_split = df.index[k]
            ax.axvspan(t_split, df.index[-1], color="#f2f2f2", zorder=0)
            ax.axvline(t_split, color="#9a9a9a", lw=0.8, ls="--")
            if ax is axes[0]:
                ax.text(t_split, 0.94, "  held-out 30 %", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=9, color="#52514e")
                ax.text(t_split, 0.94, "training region  ", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=9, color="#52514e")
        ax.set_yscale("log")
        ax.set_ylabel("radon (Bq/m³)")
        ax.grid(axis="y")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(f"Station {NAME[station]}, four models (dots = flagged samples)", x=0.01, ha="left", color=INK2)
    fig.autofmt_xdate()
    save(fig, out_dir, "negative_examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    args = ap.parse_args()
    out_dir = os.path.join(args.root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    suite = load_suite(args.root)
    met = metrics_table(suite)
    met.to_csv(os.path.join(out_dir, "method_metrics_per_station.csv"), index=False, float_format="%.4f")
    med = met.groupby("method")[["anomaly_rate", "drift", "abs_corr_err_radon"]].median().loc[METHODS]
    med.to_csv(os.path.join(out_dir, "method_metrics_median.csv"), float_format="%.4f")
    print(med.round(3).to_string())

    ref_suite = {}
    for m in METHODS:
        p = f"{args.root}/radim_suite/by_station/bozkov2017/{m}/sequence_scores.csv"
        if os.path.exists(p):
            ref_suite[("bozkov2017", m)] = read_scores(p)
    ref = metrics_table(ref_suite) if ref_suite else None
    if ref is not None:
        ref.to_csv(os.path.join(out_dir, "method_metrics_radim2017.csv"), index=False, float_format="%.4f")
    fig1_method_summary(met, out_dir, ref)
    fig2_method_agreement(suite, out_dir)
    fig3_station_consensus(suite, out_dir)
    fig4_negative_examples(suite, out_dir)


if __name__ == "__main__":
    main()
