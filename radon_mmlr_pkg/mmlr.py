from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

TIME_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
OPTIONAL_FEATURES = {"humidity_outdoor"}


@dataclass
class MmlrArtifacts:
    timestamps: pd.Index
    radon_values: np.ndarray
    predictions: np.ndarray
    reconstruction_error: np.ndarray
    anomaly_flags: np.ndarray
    coefficients: pd.DataFrame
    model: Pipeline


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    hour_float = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)

    day_of_year = df[timestamp_col].dt.dayofyear
    df["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    return df


def get_feature_columns(args) -> list[str]:
    feature_cols = list(args.features)
    if args.add_time_features:
        feature_cols.extend(TIME_FEATURES)
    return feature_cols


def resolve_feature_columns(args) -> list[str]:
    requested_features = list(args.features)
    dropped_optional = set()

    for input_path in args.input:
        header_cols = set(pd.read_csv(input_path, nrows=0).columns)
        for feature in requested_features:
            if feature not in header_cols:
                if feature in OPTIONAL_FEATURES:
                    dropped_optional.add(feature)
                else:
                    raise ValueError(f"Missing required feature column in {input_path}: {feature}")

    resolved_features = [feature for feature in requested_features if feature not in dropped_optional]
    args.features = resolved_features
    args.dropped_optional_features = sorted(dropped_optional)
    return resolved_features


def load_regression_data(args) -> pd.DataFrame:
    resolve_feature_columns(args)
    frames = []
    base_feature_cols = list(args.features)
    required_cols = [args.timestamp_col, args.arn_col] + base_feature_cols

    for input_path in args.input:
        df = pd.read_csv(input_path)
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {input_path}: {missing_cols}")

        df = df[required_cols].copy()
        df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
        for col in [args.arn_col] + base_feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required_cols).sort_values(args.timestamp_col)
        if args.add_time_features:
            df = add_time_features(df, args.timestamp_col)
        frames.append(df)

    return pd.concat(frames).sort_values(args.timestamp_col).reset_index(drop=True)


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("regression", LinearRegression()),
        ]
    )


def extract_coefficients(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    poly = model.named_steps["interactions"]
    reg = model.named_steps["regression"]
    expanded_names = poly.get_feature_names_out(feature_names)
    return pd.DataFrame(
        {
            "feature": ["intercept"] + list(expanded_names),
            "coefficient": [float(reg.intercept_)] + [float(v) for v in reg.coef_],
        }
    )


def fit_mmlr(args) -> MmlrArtifacts:
    df = load_regression_data(args)
    n_total = len(df)
    if n_total < 10:
        raise ValueError("Too few valid rows for the MMLR baseline.")

    n_train = max(2, int(n_total * args.train_fraction))
    feature_cols = get_feature_columns(args)
    X = df[feature_cols].values
    y = df[args.arn_col].values.astype(np.float64)

    transform = getattr(args, "radon_transform", "none")
    if transform == "none" and getattr(args, "log_radon", False):
        transform = "log1p"
    # Fit in transformed space: residuals stop scaling with radon level and
    # back-transformed predictions stay non-negative.
    if transform == "log1p":
        y_fit = np.log1p(np.clip(y, 0.0, None))
    elif transform == "sqrt":
        y_fit = np.sqrt(np.clip(y, 0.0, None))
    else:
        y_fit = y

    model = build_model()
    model.fit(X[:n_train], y_fit[:n_train])
    raw_predictions = model.predict(X)
    residual = y_fit - raw_predictions
    if transform == "log1p":
        predictions = np.maximum(np.expm1(raw_predictions), 0.0)
    elif transform == "sqrt":
        predictions = np.maximum(raw_predictions, 0.0) ** 2
    else:
        predictions = raw_predictions
    abs_residual = np.abs(residual)
    threshold = np.percentile(abs_residual[:n_train], args.threshold_percentile)
    anomaly_flags = abs_residual > threshold
    coefficients = extract_coefficients(model, feature_cols)

    return MmlrArtifacts(
        timestamps=pd.Index(df[args.timestamp_col]),
        radon_values=y,
        predictions=predictions,
        reconstruction_error=abs_residual,
        anomaly_flags=anomaly_flags,
        coefficients=coefficients,
        model=model,
    )
