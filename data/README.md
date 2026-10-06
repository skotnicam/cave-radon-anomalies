# Data

All timestamps are UTC. Radon concentrations are in Bq m⁻³, temperatures in °C, relative humidity in %,
pressure in hPa. Every radon record is stored as *segments*, one CSV per uninterrupted period; a new segment
starts wherever two consecutive samples are more than 3 h (stations) or 6 h (RADIM records) apart. The models
build their sliding windows inside segments only.

## stations/ — seven TSR4M stations, Bozkov Dolomite Caves, July 2022 – March 2026

Files `<station>_part<NN>.csv` for `prusacka`, `za_prahem`, `kaple`, `blativa`, `loupeznicka`, `peklo`,
`jezero` (27 segments, 192 762 hourly samples). Samples with the instrument outside the cave (internal
temperature ≥ 13 °C, plus the two following hours) are removed.

| column | meaning |
|---|---|
| `timestamp` | hour of the measurement (UTC) |
| `ARn(Bq/m3)` | radon activity concentration, one-hour moving average of the probe; the model target |
| `temperature_indoor` | cave-air temperature at the probe |
| `temperature_outdoor` | outdoor air temperature above the cave (gaps filled from ERA5-Land, calibrated) |
| `humidity_indoor` | cave-air relative humidity (used by MMLR only) |
| `pressure` | barometric pressure from the RADIM reference monitor at Jezero, interpolated (gaps filled from ERA5, calibrated) |
| `precip_24h`, `precip_72h` | 24-h and 72-h precipitation sums, ERA5-Land (used only by `experiments/09_precip.sh`) |
| `ACRn(Bq/m3)`, `sum1`–`sum4`, `HV`, `Algorit.` | raw fields of the instrument export (progeny-based estimate, channel counts, high voltage, algorithm code); not used |

The models use `ARn(Bq/m3)`, `temperature_indoor`, `temperature_outdoor`, `pressure` and the calendar
variables (sine and cosine of the hour of day and of the day of year), which the code derives from the
timestamp; the lagged-temperature variants replace `temperature_outdoor` by its value 48 h earlier.

## radim_jezero/ — RADIM reference monitor at the Jezero site, May 2017 – April 2026

`bozkov2017_part01..05.csv` (5 segments, 74 330 hourly samples; 30-min readings averaged to hours). Columns
`timestamp`, `ARn(Bq/m3)`, `pressure` (own barometer), `temperature_indoor`, `humidity_indoor`,
`temperature_outdoor` (measured from April 2021; before that ERA5-Land calibrated to the sensor, 32 095 of
the 74 330 hours). 311 hours in four episodes with the instrument outside the cave are removed.

`jezero_period/bozkov2017jp_part*.csv`: the same record restricted to the period in which the Jezero station
runs alongside it (March 2024 – March 2026), for the two-instrument comparison; `train_fraction.txt` is the
training fraction that ends on the same day (21 August 2025) as the 70 % training region of the station.

## ochtinska/ — RADIM monitor, Ochtinská Aragonite Cave, June 2016 – February 2024

`ochtinska_part01..04.csv` (4 segments, 50 396 hourly samples), same columns as the Bozkov RADIM record. Only
periods with measured outdoor temperature are included (June 2018 – October 2019 is left out).

## microseisms/daily_values.csv — seismic station OKC (Ostrava–Krásné Pole), 2016–2026

One row per day. For every day one night hour of the vertical component (23:00–24:00 CET throughout the
year, instrument response removed) was checked visually; when it contained a transient, a window shifted by
up to ±3 h was used (`status`, `offset_h`, `window_start_utc`). `rms_total_m_s` is the root-mean-square
velocity in 0.05–0.30 Hz (the daily intensity), `rms_primary_m_s` and `rms_secondary_m_s` the sub-bands
0.05–0.10 Hz and 0.10–0.30 Hz; the remaining columns are quality diagnostics of the visual check. 3 403 of
the 3 826 days have a value.

## catalogues/ — earthquake catalogue extracts

| file | content |
|---|---|
| `emsc_catalog_2016_2026.csv` | EMSC, Central Europe, M ≥ 1.8 (`time, lat, lon, depth_km, mag, magtype`) |
| `isc_slovakia_2019_2024.csv` | ISC bulletin, Slovakia and surroundings, M ≥ 1.5 |
| `usgs_global_M55_2017_2022.csv`, `usgs_global_M6_2022_2026.csv` | USGS ComCat, worldwide (native USGS columns) |
| `czech_bulletin_2016_2026.csv` | regional bulletin of the Czech Regional Seismic Network (Institute of Geophysics, CAS); `Type` T = tectonic, M = mining-induced; `mag` is the preferred magnitude |

## meteo/

`openmeteo_bozkov_daily.csv`: ERA5-Land daily series for Bozkov from the Open-Meteo archive (precipitation
sum, surface pressure, 2-m temperature), used for the rainfall analyses. `radim_jezero_pressure_hourly.csv`:
hourly mean of the barometer of the reference monitor, May 2017 – May 2026, used for the pressure-low
analysis.
