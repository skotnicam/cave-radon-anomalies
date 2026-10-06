import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


TIME_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
OPTIONAL_BASE_FEATURES = {"humidity_outdoor"}
LAGGED_OUTDOOR_TEMP_COL = "temperature_outdoor_lagged"
LOG_RADON_COL = "arn_log1p"
SQRT_RADON_COL = "arn_sqrt"
TRANSFORMED_RADON_COLS = {"log1p": LOG_RADON_COL, "sqrt": SQRT_RADON_COL}


def radon_transform(args) -> str | None:
    """Resolve the radon transform: --radon-transform wins, --log-radon is a legacy alias."""
    transform = getattr(args, "radon_transform", None)
    if transform in (None, "none") and getattr(args, "log_radon", False):
        transform = "log1p"
    return None if transform in (None, "none") else transform


def radon_model_col(args) -> str:
    """Column the models should use for radon: raw, or the transformed variant."""
    transform = radon_transform(args)
    return TRANSFORMED_RADON_COLS[transform] if transform else args.arn_col


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    X: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray | None
    X_train_total: np.ndarray
    seq_end_times: pd.Index


@dataclass
class PreparedSupervisedData:
    df: pd.DataFrame
    input_feature_cols: list[str]
    target_feature_cols: list[str]
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    X: np.ndarray
    y: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray | None
    y_val: np.ndarray | None
    X_train_total: np.ndarray
    y_train_total: np.ndarray
    sample_end_times: pd.Index
    sample_radon_values: np.ndarray


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    df = df.set_index(timestamp_col)

    hour_float = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)

    day_of_year = df.index.dayofyear
    df["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    return df


def create_sequences(data: np.ndarray, window: int) -> np.ndarray:
    sequences = []
    for i in range(len(data) - window + 1):
        sequences.append(data[i:i + window])
    return np.asarray(sequences, dtype=np.float32)


def chronological_train_val_split(X: np.ndarray, train_fraction: float, validation_size: float, train_region: str = "head"):
    n_total = len(X)
    n_train_total = max(1, int(n_total * train_fraction))
    # "head": earliest fraction is the training region (default); "tail": latest fraction.
    X_train_total = X[-n_train_total:] if train_region == "tail" else X[:n_train_total]

    n_val = max(1, int(len(X_train_total) * validation_size))
    if len(X_train_total) - n_val < 1:
        n_val = max(0, len(X_train_total) - 1)

    if n_val == 0:
        X_train = X_train_total
        X_val = None
    else:
        X_train = X_train_total[:-n_val]
        X_val = X_train_total[-n_val:]

    return X_train, X_val, X_train_total


def chronological_train_val_split_xy(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float,
    validation_size: float,
    train_region: str = "head",
):
    X_train, X_val, X_train_total = chronological_train_val_split(
        X, train_fraction=train_fraction, validation_size=validation_size, train_region=train_region
    )
    n_train_total = len(X_train_total)
    n_train = len(X_train)
    offset = len(X) - n_train_total if train_region == "tail" else 0
    y_train_total = y[offset:offset + n_train_total]
    y_train = y[offset:offset + n_train]
    y_val = None if X_val is None else y[offset + n_train:offset + n_train_total]
    return X_train, X_val, X_train_total, y_train, y_val, y_train_total


def fit_standardization_stats(train_array: np.ndarray):
    mean = train_array.mean(axis=0)
    std = train_array.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_array(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((array - mean) / std).astype(np.float32)


def resolve_base_feature_cols(args) -> list[str]:
    requested_features = list(args.features)
    dropped_optional = set()

    for input_path in list(args.input):
        header_cols = set(pd.read_csv(input_path, nrows=0).columns)
        for feature in requested_features:
            if feature not in header_cols:
                if feature in OPTIONAL_BASE_FEATURES:
                    dropped_optional.add(feature)
                else:
                    raise ValueError(f"Missing required feature column in {input_path}: {feature}")

    resolved_features = [feature for feature in requested_features if feature not in dropped_optional]
    args.features = resolved_features
    args.dropped_optional_features = sorted(dropped_optional)
    return resolved_features


def ensure_required_columns(df: pd.DataFrame, required_cols: list[str], input_path: str):
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {input_path}: {missing_cols}")


def coerce_numeric_columns(df: pd.DataFrame, cols: list[str], input_path: str) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = numeric.isna() & df[col].notna()
        if invalid_mask.any():
            examples = df.loc[invalid_mask, col].astype(str).head(3).tolist()
            raise ValueError(f"Non-numeric values in {col} for {input_path}: {examples}")
        df[col] = numeric
    return df


def prepare_single_file_dataframe(args, input_path: str, required_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    computed_cols = {"temp_diff", LAGGED_OUTDOOR_TEMP_COL, LOG_RADON_COL, SQRT_RADON_COL}
    raw_required_cols = [
        c for c in required_cols if c not in TIME_FEATURES and c not in computed_cols
    ]
    ensure_required_columns(df, raw_required_cols, input_path)
    df = add_time_features(df, args.timestamp_col)

    transform = radon_transform(args)
    if transform:
        df = coerce_numeric_columns(df, [args.arn_col], input_path)
        # Negative readings are sensor artifacts; clip so the transforms stay defined.
        clipped = df[args.arn_col].clip(lower=0.0)
        if transform == "log1p":
            df[LOG_RADON_COL] = np.log1p(clipped)
        else:
            # Variance-stabilizing for counting noise: between raw (error tracks
            # amplitude) and log (error tracks low-level relative noise).
            df[SQRT_RADON_COL] = np.sqrt(clipped)

    if getattr(args, "temp_outdoor_lag_hours", 0) > 0:
        if "temperature_outdoor" not in df.columns:
            raise ValueError(
                f"--temp-outdoor-lag-hours requires temperature_outdoor in {input_path}."
            )
        df = coerce_numeric_columns(df, ["temperature_outdoor"], input_path)
        df[LAGGED_OUTDOOR_TEMP_COL] = df["temperature_outdoor"].shift(args.temp_outdoor_lag_hours)

    if args.add_temp_diff:
        if "temperature_indoor" not in df.columns or "temperature_outdoor" not in df.columns:
            raise ValueError(
                f"--add-temp-diff requires temperature_indoor and temperature_outdoor in {input_path}."
            )
        df = coerce_numeric_columns(df, ["temperature_indoor", "temperature_outdoor"], input_path)
        df["temp_diff"] = df["temperature_indoor"] - df["temperature_outdoor"]

    return df


def load_ordered_dataframes(args, required_cols: list[str]) -> list[pd.DataFrame]:
    dataframes = [
        prepare_single_file_dataframe(args, input_path, required_cols)
        for input_path in list(args.input)
    ]
    dataframes.sort(key=lambda df: df.index.min())
    return dataframes


def clean_frame_for_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = model_df.notna().all(axis=1)
    clean_df = df.loc[valid_mask].copy()
    clean_features = model_df.loc[valid_mask].copy()
    return clean_df, clean_features


def combine_clean_frames(
    dataframes: list[pd.DataFrame],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_frames = []
    cleaned_features = []

    for df in dataframes:
        clean_df, clean_feature_df = clean_frame_for_features(df, feature_cols)
        if not clean_feature_df.empty:
            cleaned_frames.append(clean_df)
            cleaned_features.append(clean_feature_df)

    if not cleaned_features:
        raise ValueError("No valid rows remain after cleaning the provided files.")

    combined_df = pd.concat(cleaned_frames).sort_index()
    combined_feature_df = pd.concat(cleaned_features).sort_index()
    return combined_df, combined_feature_df


def build_reconstruction_samples_per_file(
    dataframes: list[pd.DataFrame],
    input_feature_cols: list[str],
    target_feature_cols: list[str],
    window: int,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    arn_col: str,
):
    X_parts = []
    y_parts = []
    time_parts = []
    radon_parts = []
    cleaned_frames = []

    required_cols = sorted(set(input_feature_cols) | set(target_feature_cols) | {arn_col})
    for df in dataframes:
        clean_df, clean_feature_df = clean_frame_for_features(df, required_cols)
        if len(clean_feature_df) < window:
            continue

        input_values = standardize_array(
            clean_feature_df[input_feature_cols].values.astype(np.float32), input_mean, input_std
        )
        target_values = standardize_array(
            clean_feature_df[target_feature_cols].values.astype(np.float32), target_mean, target_std
        )

        X_parts.append(create_sequences(input_values, window))
        y_parts.append(create_sequences(target_values, window))
        time_parts.append(clean_feature_df.index[window - 1:])
        radon_parts.append(clean_df[arn_col].iloc[window - 1:].values.astype(np.float32))
        cleaned_frames.append(clean_df)

    if not X_parts:
        raise ValueError(
            f"No reconstruction samples could be created. "
            f"Each file needs at least window={window} valid rows."
        )

    return (
        pd.concat(cleaned_frames).sort_index(),
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        pd.Index(np.concatenate([idx.to_numpy() for idx in time_parts])),
        np.concatenate(radon_parts, axis=0),
    )


def build_forecast_samples_per_file(
    dataframes: list[pd.DataFrame],
    input_feature_cols: list[str],
    target_feature_cols: list[str],
    window: int,
    horizon: int,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    arn_col: str,
):
    X_parts = []
    y_parts = []
    time_parts = []
    radon_parts = []
    cleaned_frames = []

    required_cols = sorted(set(input_feature_cols) | set(target_feature_cols) | {arn_col})
    for df in dataframes:
        clean_df, clean_feature_df = clean_frame_for_features(df, required_cols)
        min_rows = window + horizon
        if len(clean_feature_df) < min_rows:
            continue

        input_values = standardize_array(
            clean_feature_df[input_feature_cols].values.astype(np.float32), input_mean, input_std
        )
        target_values = standardize_array(
            clean_feature_df[target_feature_cols].values.astype(np.float32), target_mean, target_std
        )

        file_X = []
        file_y = []
        file_times = []
        file_radon = []
        for start in range(len(clean_feature_df) - window - horizon + 1):
            x_end = start + window
            y_end = x_end + horizon
            file_X.append(input_values[start:x_end])
            file_y.append(target_values[x_end:y_end])
            file_times.append(clean_feature_df.index[y_end - 1])
            file_radon.append(float(clean_df[arn_col].iloc[y_end - 1]))

        X_parts.append(np.asarray(file_X, dtype=np.float32))
        y_parts.append(np.asarray(file_y, dtype=np.float32))
        time_parts.append(np.asarray(file_times))
        radon_parts.append(np.asarray(file_radon, dtype=np.float32))
        cleaned_frames.append(clean_df)

    if not X_parts:
        raise ValueError(
            f"No forecast samples could be created. "
            f"Each file needs at least window + forecast_horizon={window + horizon} valid rows."
        )

    return (
        pd.concat(cleaned_frames).sort_index(),
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        pd.Index(np.concatenate(time_parts)),
        np.concatenate(radon_parts, axis=0),
    )


def build_model_feature_cols(args) -> list[str]:
    feature_cols = []
    for feature in list(args.features):
        if (
            feature == "temperature_outdoor"
            and getattr(args, "temp_outdoor_lag_hours", 0) > 0
            and getattr(args, "replace_temp_outdoor_with_lag", False)
        ):
            feature_cols.append(LAGGED_OUTDOOR_TEMP_COL)
        elif feature == args.arn_col and radon_transform(args):
            feature_cols.append(radon_model_col(args))
        else:
            feature_cols.append(feature)

    if (
        getattr(args, "temp_outdoor_lag_hours", 0) > 0
        and not getattr(args, "replace_temp_outdoor_with_lag", False)
        and LAGGED_OUTDOOR_TEMP_COL not in feature_cols
    ):
        feature_cols.append(LAGGED_OUTDOOR_TEMP_COL)

    feature_cols += TIME_FEATURES
    if args.add_temp_diff:
        feature_cols.append("temp_diff")
    return feature_cols


def _train_rows_slice(features: pd.DataFrame, n_rows_for_train: int, args) -> pd.DataFrame:
    """Rows used for standardization statistics: earliest (head) or latest (tail) part of the record."""
    if getattr(args, "train_region", "head") == "tail":
        return features.iloc[-n_rows_for_train:]
    return features.iloc[:n_rows_for_train]


def validate_total_rows(total_rows: int, minimum_rows: int, label: str):
    if total_rows < minimum_rows:
        raise ValueError(
            f"Not enough valid rows after cleaning for {label}: {total_rows} rows. "
            f"Need at least {minimum_rows}."
        )


def prepare_data(args) -> PreparedData:
    resolve_base_feature_cols(args)
    feature_cols = build_model_feature_cols(args)
    required_cols = sorted(set(feature_cols) | {args.timestamp_col, args.arn_col})
    dataframes = load_ordered_dataframes(args, required_cols)
    combined_df, combined_model_df = combine_clean_frames(dataframes, feature_cols)

    validate_total_rows(len(combined_model_df), args.window * 3, "reconstruction")

    n_rows_for_train = max(args.window, int(len(combined_model_df) * args.train_fraction))
    train_rows = _train_rows_slice(combined_model_df, n_rows_for_train, args).values.astype(np.float32)
    mean, std = fit_standardization_stats(train_rows)

    df, X, _, seq_end_times, _ = build_reconstruction_samples_per_file(
        dataframes=dataframes,
        input_feature_cols=feature_cols,
        target_feature_cols=feature_cols,
        window=args.window,
        input_mean=mean,
        input_std=std,
        target_mean=mean,
        target_std=std,
        arn_col=args.arn_col,
    )

    if len(X) < 10:
        raise ValueError("Too few sequences created. Reduce --window or provide more data.")

    X_train, X_val, X_train_total = chronological_train_val_split(
        X, train_fraction=args.train_fraction, validation_size=args.validation_size,
        train_region=getattr(args, "train_region", "head"),
    )

    if len(X_train) < 1:
        raise ValueError("Training set is empty after split.")

    return PreparedData(
        df=df,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        X=X,
        X_train=X_train,
        X_val=X_val,
        X_train_total=X_train_total,
        seq_end_times=seq_end_times,
    )


def prepare_reconstruction_supervised_data(
    args,
    input_feature_cols: list[str],
    target_feature_cols: list[str],
) -> PreparedSupervisedData:
    resolve_base_feature_cols(args)
    required_cols = sorted(
        set(input_feature_cols) | set(target_feature_cols) | {args.timestamp_col, args.arn_col}
    )
    dataframes = load_ordered_dataframes(args, required_cols)

    combined_input_df, combined_input_features = combine_clean_frames(dataframes, input_feature_cols)
    _, combined_target_features = combine_clean_frames(dataframes, target_feature_cols)

    validate_total_rows(len(combined_input_features), args.window * 3, "reconstruction")

    n_rows_for_train = max(args.window, int(len(combined_input_features) * args.train_fraction))
    input_train_rows = _train_rows_slice(combined_input_features, n_rows_for_train, args).values.astype(np.float32)
    target_train_rows = _train_rows_slice(combined_target_features, n_rows_for_train, args).values.astype(np.float32)
    input_mean, input_std = fit_standardization_stats(input_train_rows)
    target_mean, target_std = fit_standardization_stats(target_train_rows)

    df, X, y, sample_end_times, sample_radon_values = build_reconstruction_samples_per_file(
        dataframes=dataframes,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
        window=args.window,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        arn_col=args.arn_col,
    )

    if len(X) < 10:
        raise ValueError("Too few sequences created. Reduce --window or provide more data.")

    X_train, X_val, X_train_total, y_train, y_val, y_train_total = chronological_train_val_split_xy(
        X,
        y,
        train_fraction=args.train_fraction,
        validation_size=args.validation_size,
        train_region=getattr(args, "train_region", "head"),
    )

    if len(X_train) < 1:
        raise ValueError("Training set is empty after split.")

    return PreparedSupervisedData(
        df=combined_input_df,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        X=X,
        y=y,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_train_total=X_train_total,
        y_train_total=y_train_total,
        sample_end_times=sample_end_times,
        sample_radon_values=sample_radon_values,
    )


def prepare_forecast_supervised_data(
    args,
    input_feature_cols: list[str],
    target_feature_cols: list[str],
) -> PreparedSupervisedData:
    resolve_base_feature_cols(args)
    required_cols = sorted(
        set(input_feature_cols) | set(target_feature_cols) | {args.timestamp_col, args.arn_col}
    )
    dataframes = load_ordered_dataframes(args, required_cols)

    combined_input_df, combined_input_features = combine_clean_frames(dataframes, input_feature_cols)
    _, combined_target_features = combine_clean_frames(dataframes, target_feature_cols)

    min_rows = args.window + args.forecast_horizon + 5
    validate_total_rows(len(combined_input_features), min_rows, "forecasting")

    n_rows_for_train = max(args.window + args.forecast_horizon, int(len(combined_input_features) * args.train_fraction))
    input_train_rows = _train_rows_slice(combined_input_features, n_rows_for_train, args).values.astype(np.float32)
    target_train_rows = _train_rows_slice(combined_target_features, n_rows_for_train, args).values.astype(np.float32)
    input_mean, input_std = fit_standardization_stats(input_train_rows)
    target_mean, target_std = fit_standardization_stats(target_train_rows)

    df, X, y, sample_end_times, sample_radon_values = build_forecast_samples_per_file(
        dataframes=dataframes,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
        window=args.window,
        horizon=args.forecast_horizon,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        arn_col=args.arn_col,
    )

    if len(X) < 10:
        raise ValueError("Too few forecast samples created. Reduce --window/--forecast-horizon or provide more data.")

    X_train, X_val, X_train_total, y_train, y_val, y_train_total = chronological_train_val_split_xy(
        X,
        y,
        train_fraction=args.train_fraction,
        validation_size=args.validation_size,
        train_region=getattr(args, "train_region", "head"),
    )

    if len(X_train) < 1:
        raise ValueError("Training set is empty after split.")

    return PreparedSupervisedData(
        df=combined_input_df,
        input_feature_cols=input_feature_cols,
        target_feature_cols=target_feature_cols,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        X=X,
        y=y,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_train_total=X_train_total,
        y_train_total=y_train_total,
        sample_end_times=sample_end_times,
        sample_radon_values=sample_radon_values,
    )
