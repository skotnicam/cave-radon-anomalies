#!/usr/bin/env python3
"""Training and validation loss per epoch for the eight neural methods (seven Bozkov stations).
One panel per method; thin lines = stations (train solid, validation dashed), thick = median over stations.
Output: <root>/figures/training_curves.{png,pdf} + training_curves.csv (final train/val loss, epochs, val/train ratio).
Usage: python analysis/training_curves.py [runs_root]
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
BASE = os.path.join(ROOT, "main_suite", "by_station")
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
STATIONS = ["prusacka", "za_prahem", "kaple", "blativa", "loupeznicka", "peklo", "jezero"]
METHODS = [("reconstruct_radon", "LSTM reconstruction"), ("reconstruct_delayed_temp", "LSTM, delayed temperature"),
           ("transformer_reconstruct_radon", "Transformer reconstruction"), ("transformer_reconstruct_delayed_temp", "Transformer, delayed temp."),
           ("conv_reconstruct_radon", "Conv reconstruction"), ("forecast", "LSTM forecast"),
           ("conv_forecast", "Conv forecast"), ("meteo_to_radon", "Meteo → radon (LSTM)")]
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#0b0b0b"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.titleweight": "bold"})

fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharex=True)
rows = []
HIGHLIGHT = {"reconstruct_radon", "meteo_to_radon"}  # contrasted in the text (panels a and h)
for k, (ax, (m, title)) in enumerate(zip(axes.ravel(), METHODS)):
    title = f"({'abcdefgh'[k]}) {title}"
    hist = {}
    for st in STATIONS:
        p = f"{BASE}/{st}/{m}/training_history.csv"
        if os.path.exists(p):
            h = pd.read_csv(p); h.index = np.arange(1, len(h) + 1); hist[st] = h
            rows.append(dict(method=m, station=st, epochs=len(h), final_train=h.train_loss.iloc[-1], final_val=h.val_loss.iloc[-1],
                             best_val=h.val_loss.min(), best_epoch=int(h.val_loss.idxmin()), val_over_train=h.val_loss.iloc[-1] / h.train_loss.iloc[-1]))
            ax.plot(h.index, h.train_loss, color=BLUE, lw=0.8, alpha=0.45)
            ax.plot(h.index, h.val_loss, color=ORANGE, lw=0.8, alpha=0.45, ls="--")
    if hist:
        T = pd.concat({st: h.train_loss for st, h in hist.items()}, axis=1).median(axis=1)
        V = pd.concat({st: h.val_loss for st, h in hist.items()}, axis=1).median(axis=1)
        ax.plot(T.index, T, color=BLUE, lw=2.2, label="training loss")
        ax.plot(V.index, V, color=ORANGE, lw=2.2, ls="--", label="validation loss")
    ax.set_yscale("log"); ax.set_title(title, loc="left")
    if m in HIGHLIGHT:  # thin coloured frame around the two panels compared in the text
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#4a3aa7"); sp.set_linewidth(1.6)
    else:
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
for ax in axes[1]: ax.set_xlabel("epoch")
for ax in axes[:, 0]: ax.set_ylabel("MSE (standardized units)")
axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Training (solid) and validation (dashed) loss; thin lines = the seven stations, thick = median. Early stopping on the validation loss (patience 10).",
             x=0.01, ha="left", fontsize=10.5, color="#52514e")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"training_curves.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "training_curves.csv"), index=False, float_format="%.4f")
print(R.groupby("method")[["epochs", "final_train", "final_val", "val_over_train"]].median().loc[[m for m, _ in METHODS]].round(3).to_string())
print("wrote training_curves")
