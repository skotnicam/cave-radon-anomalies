from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.svm import SVR


@dataclass
class EmdSvrArtifacts:
    timestamps: pd.Index
    radon_values: np.ndarray
    predictions: np.ndarray
    upper_bound: np.ndarray
    lower_bound: np.ndarray
    positive_anomaly: np.ndarray
    negative_anomaly: np.ndarray
    anomaly_flags: np.ndarray
    reconstruction_error: np.ndarray
    imfs: np.ndarray


def parse_svr_gamma(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def load_radon_series(args) -> tuple[pd.Index, np.ndarray]:
    frames = []
    for input_path in args.input:
        df = pd.read_csv(input_path)
        required_cols = [args.timestamp_col, args.arn_col]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {input_path}: {missing_cols}")

        df = df[[args.timestamp_col, args.arn_col]].copy()
        df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
        df[args.arn_col] = pd.to_numeric(df[args.arn_col], errors="coerce")
        df = df.dropna(subset=[args.timestamp_col, args.arn_col]).sort_values(args.timestamp_col)
        frames.append(df)

    combined = pd.concat(frames).sort_values(args.timestamp_col)
    return pd.Index(combined[args.timestamp_col]), combined[args.arn_col].values.astype(np.float64)


def _spline_envelope(indices: np.ndarray, values: np.ndarray, n_points: int) -> np.ndarray:
    unique_indices, unique_pos = np.unique(indices, return_index=True)
    unique_values = values[unique_pos]
    if len(unique_indices) < 2:
        raise ValueError("Not enough unique extrema for spline envelope.")
    spline = CubicSpline(unique_indices, unique_values, bc_type="natural")
    return spline(np.arange(n_points))


def emd_decompose(signal: np.ndarray, max_imfs: int = 20, max_sift_iterations: int = 200) -> np.ndarray:
    c = signal.astype(np.float64).copy()
    n = len(signal)
    imfs = []

    while len(imfs) < max_imfs:
        h = c.copy()
        sd = 1.0
        maxmin = np.array([], dtype=int)
        sift_iterations = 0

        while sd > 0.3 and sift_iterations < max_sift_iterations:
            sift_iterations += 1
            d = np.diff(h)
            maxmin_list = []
            for i in range(n - 2):
                if d[i] == 0:
                    maxmin_list.append(i)
                elif np.sign(d[i]) != np.sign(d[i + 1]):
                    maxmin_list.append(i + 1)

            maxmin = np.asarray(maxmin_list, dtype=int)
            if maxmin.size < 2:
                break

            if maxmin[0] > maxmin[1]:
                maxes = maxmin[0::2]
                mins = maxmin[1::2]
            else:
                maxes = maxmin[1::2]
                mins = maxmin[0::2]

            maxes = np.concatenate(([0], maxes, [n - 1]))
            mins = np.concatenate(([0], mins, [n - 1]))

            try:
                maxenv = _spline_envelope(maxes, h[maxes], n)
                try:
                    minenv = _spline_envelope(mins, h[mins], n)
                    m = (maxenv + minenv) / 2.0
                except ValueError:
                    m = maxenv
            except ValueError:
                minenv = _spline_envelope(mins, h[mins], n)
                try:
                    maxenv = _spline_envelope(maxes, h[maxes], n)
                    m = (maxenv + minenv) / 2.0
                except ValueError:
                    m = minenv

            prevh = h.copy()
            h = h - m
            if np.allclose(prevh, h):
                break
            eps = 1e-7
            sd = np.sum(((prevh - h) ** 2) / (prevh**2 + eps))

        imfs.append(h.copy())
        if maxmin.size < 2:
            break
        c = c - h

    return np.column_stack(imfs)


def fit_emd_svr(args) -> EmdSvrArtifacts:
    timestamps, x = load_radon_series(args)
    imfs = emd_decompose(x)
    gamma = parse_svr_gamma(args.svr_gamma)

    yfit_columns = []
    for i in range(imfs.shape[1]):
        svr = SVR(kernel="rbf", C=args.svr_c, epsilon=args.svr_epsilon, gamma=gamma)
        feature = imfs[:, i].reshape(-1, 1)
        svr.fit(feature, x)
        yfit_columns.append(svr.predict(feature))

    yfit_matrix = np.column_stack(yfit_columns)
    meta_svr = SVR(kernel="rbf", C=args.svr_c, epsilon=args.svr_epsilon, gamma=gamma)
    meta_svr.fit(yfit_matrix, x)
    predictions = meta_svr.predict(yfit_matrix)

    mse = float(np.mean((x - predictions) ** 2))
    band = 1.96 * np.sqrt(abs(mse))
    upper_bound = predictions + band
    lower_bound = predictions - band

    positive_anomaly = np.maximum(0.0, x - upper_bound)
    negative_anomaly = np.maximum(0.0, lower_bound - x)
    anomaly_flags = (positive_anomaly > 0) | (negative_anomaly > 0)
    reconstruction_error = np.abs(x - predictions)

    return EmdSvrArtifacts(
        timestamps=timestamps,
        radon_values=x,
        predictions=predictions,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        positive_anomaly=positive_anomaly,
        negative_anomaly=negative_anomaly,
        anomaly_flags=anomaly_flags,
        reconstruction_error=reconstruction_error,
        imfs=imfs,
    )
