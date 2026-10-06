# Cave radon anomalies: data and code

Data and code for the paper

> O. Kořistka, M. Skotnica, *Neural-network approaches to radon anomaly detection in a cave monitoring network: comparison with classical methods and tests of association with seismicity*, submitted to Radiation Physics and Chemistry (2026).

Hourly radon records from seven points of the Bozkov Dolomite Caves (Czechia, 2022–2026), the nine-year
record of a reference monitor at one of them (2017–2026) and the Ochtinská Aragonite Cave (Slovakia,
2016–2024) are scored by ten unsupervised anomaly detectors under one decision rule: LSTM, convolutional
and Transformer autoencoders and forecasters, a regression from meteorological covariates, multiple
meteorological linear regression (MMLR) and empirical mode decomposition with support-vector regression
(EMD–SVR). The repository contains the prepared data, the models, the experiment scripts and the
analysis scripts that produce every figure and table of the paper.

## Layout

```
data/                 prepared input data (see data/README.md)
  stations/           7 TSR4M stations, hourly, 2022-2026, one CSV per uninterrupted segment
  radim_jezero/       RADIM reference monitor at the Jezero site, 2017-2026 (+ the period shared with the station)
  ochtinska/          RADIM monitor in the Ochtinska Aragonite Cave, 2016-2024
  microseisms/        daily microseism intensity at seismic station OKC, 2016-2026
  catalogues/         earthquake catalogue extracts (EMSC, ISC, USGS, Czech regional bulletin)
  meteo/              ERA5-Land daily precipitation for Bozkov; hourly barometer of the reference monitor
radon_lstm_ae_torch_pkg/   neural models (LSTM, Conv1D, Transformer; reconstruction, forecast, meteo -> radon)
radon_mmlr_pkg/            MMLR baseline
radon_emd_svr_pkg/         EMD-SVR baseline
radon_*.py                 command-line entry points of the methods (one per method) and of the transfer model
experiments/               the experiments of the paper, numbered; run_all.sh runs them in order
analysis/                  figures and tables from the experiment outputs; make_figures.sh runs them all
figures/                   the figures of the paper as produced by make_figures.sh
```

## Installation

Python 3.10 or newer with NumPy, pandas, SciPy, scikit-learn, PyTorch (CPU is enough) and Matplotlib:

```
pip install -r requirements.txt
```

## Running one method

Every method reads one or more segment CSVs of one record and writes `sequence_scores.csv` (timestamp,
reconstruction error, threshold, normalized score `relative_error` = error / threshold, flag `is_anomaly`,
radon) plus `training_history.csv`, `anomalies.csv` and diagnostic plots into the output directory:

```
python radon_reconstruct_from_window.py --input data/stations/kaple_part*.csv --output-dir out/kaple_lstm \
    --window 24 --epochs 40 --batch-size 32 --hidden-size 64 --latent-size 16 --learning-rate 1e-3 \
    --min-epochs 10 --early-stopping-patience 10 --log-radon --grad-clip 1.0 --weight-decay 1e-4
python radon_mmlr.py    --input data/stations/kaple_part*.csv --output-dir out/kaple_mmlr --add-time-features --log-radon
python radon_emd_svr.py --input data/stations/kaple_part*.csv --output-dir out/kaple_emd
```

`experiments/01_main_suite.sh` lists the exact arguments of all ten methods (Table 2 of the paper). The
common decision rule: the first 70 % of each record is the training region, its last tenth is held out for
early stopping, the threshold is the 95th percentile of the training-region error, and samples with an error
above it are flagged. Radon enters the neural models and MMLR as log(1 + radon); EMD–SVR works on the raw
concentration and, as published, on the whole series with a ±1.96·RMSE band. Windows never bridge a gap
between segments. The seed is fixed (42) and training runs on the CPU, so a rerun reproduces the published
runs up to floating-point differences between library versions.

## Reproducing the paper

```
experiments/run_all.sh runs        # all model runs into runs/, then the figures
./make_figures.sh runs             # figures and tables only (needs the runs they are based on)
```

| Script | What it trains | Used in |
|---|---|---|
| `01_main_suite.sh` | ten methods × seven stations (70 runs, ~75 min on a laptop CPU) | Sections 4.1–4.3, 4.5, Appendices A–C, E |
| `02_radim_suite.sh` | ten methods on the 2017–2026 reference monitor | Sections 4.1, 4.3, 4.5 |
| `03_ochtinska.sh` | LSTM, Transformer, MMLR on Ochtinská | Appendix D |
| `04_station_transfer.sh` | LSTM trained on one station, scored on another (42 pairs) | Section 4.4 |
| `05_multistation.sh` | LSTM trained on three stations | Section 4.4, Table 4 |
| `06_cave_transfer.sh` | LSTM between the two RADIM records | Section 4.4, Appendix D |
| `07_probe_comparison.sh` | seeds and training periods at the Jezero site | Section 4.5, Appendix F |
| `08_radim_leaveout.sh` | models trained without ±60 days around five earthquakes | Section 4.3 |
| `09_precip.sh` | precipitation sums as an extra input | Section 4.3 |
| `10_meteo_latent.sh` | meteo → radon with a larger latent space | Section 4.1 |

Each script takes the runs root as its only argument (default `runs`), accepts `RESUME=1` to skip finished
runs and writes its outputs under `runs/<experiment>/`. `make_figures.sh` lists next to every analysis which
experiments it needs; the per-figure PDFs land in `runs/figures/` and are copied to `figures/`. The
seismicity tests and the microseism comparison need only the data in `data/` and the runs of 01–03.

## Data

`data/README.md` describes every file and column. Timestamps are UTC. The radon records were segmented at
gaps longer than 3 h (stations) or 6 h (RADIM), samples recorded with the instruments outside the cave were
removed, barometric pressure of the stations comes from the barometer of the reference monitor, and missing
outdoor temperature and pressure were filled from ERA5-Land and ERA5 (Open-Meteo archive) calibrated to the
local sensors. The preparation of the raw instrument exports is not part of this repository.

## License

Code: MIT License (see `LICENSE`). Data in `data/stations`, `data/radim_jezero`, `data/ochtinska`,
`data/microseisms` and `data/meteo`: CC BY 4.0, please cite the paper. The catalogue extracts in
`data/catalogues` remain under the terms of their providers (EMSC, ISC, USGS, Institute of Geophysics of the
Czech Academy of Sciences); they are included for reproducibility only.

## Acknowledgements

Cave Administration of the Czech Republic (access, cave map and the record of cave operations); ERA5 and
ERA5-Land (ECMWF, Copernicus Climate Change Service) via Open-Meteo; EMSC, USGS, ISC and the Czech Regional
Seismic Network bulletin; Jana Rušajová (microseisms). M. Skotnica was supported by the Programme to Support
Prospective Human Resources – Postdoctoral Fellows of the Czech Academy of Sciences, reg. no. L100862601.
