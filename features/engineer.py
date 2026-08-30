import logging
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENSOR_COLS: List[str] = ["NO3", "Turbidity", "Temp", "Conductivity"]
LAG_WINDOWS: List[int] = [1, 3, 6, 12, 24]
ROLLING_WINDOWS: List[int] = [3, 6, 12, 24]


def add_lag_features(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Generate discrete autoregressive lag variables."""
    target_cols = cols or SENSOR_COLS
    df_out = df.copy()
    for col in target_cols:
        if col not in df_out.columns:
            continue
        for lag in LAG_WINDOWS:
            df_out[f"{col}_lag_{lag}h"] = df_out[col].shift(lag)
    return df_out


def add_rolling_features(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Calculate rolling statistical moments across historical temporal windows."""
    target_cols = cols or SENSOR_COLS
    df_out = df.copy()
    for col in target_cols:
        if col not in df_out.columns:
            continue
        for window in ROLLING_WINDOWS:
            roll = df_out[col].rolling(window=window, min_periods=1)
            df_out[f"{col}_roll_mean_{window}h"] = roll.mean()
            df_out[f"{col}_roll_std_{window}h"] = roll.std().fillna(0.0)
            df_out[f"{col}_roll_min_{window}h"] = roll.min()
            df_out[f"{col}_roll_max_{window}h"] = roll.max()
    return df_out


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Project diurnal and seasonal indices onto orthogonal sine and cosine bases."""
    df_out = df.copy()
    if "hour" in df_out.columns:
        df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour"] / 24.0)
        df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour"] / 24.0)
    if "day_of_week" in df_out.columns:
        df_out["dow_sin"] = np.sin(2 * np.pi * df_out["day_of_week"] / 7.0)
        df_out["dow_cos"] = np.cos(2 * np.pi * df_out["day_of_week"] / 7.0)
    if "month" in df_out.columns:
        df_out["month_sin"] = np.sin(2 * np.pi * df_out["month"] / 12.0)
        df_out["month_cos"] = np.cos(2 * np.pi * df_out["month"] / 12.0)
    return df_out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical domain proxies for sediment dynamics and agricultural runoff."""
    df_out = df.copy()

    for col in ["NO3", "Turbidity", "Temp"]:
        if col in df_out.columns:
            df_out[f"{col}_delta_1h"] = df_out[col].diff(1).fillna(0.0)

    if "Turbidity" in df_out.columns and "Temp" in df_out.columns:
        df_out["turb_temp_stress"] = (df_out["Turbidity"] * df_out["Temp"]) / 100.0

    if "NO3" in df_out.columns and "Conductivity" in df_out.columns:
        df_out["no3_cond_ratio"] = df_out["NO3"] / (df_out["Conductivity"].clip(lower=1e-3))

    if "NO3" in df_out.columns and "Turbidity" in df_out.columns:
        df_out["no3_turb_corr_6h"] = (
            df_out["NO3"].rolling(window=6, min_periods=3).corr(df_out["Turbidity"]).fillna(0.0)
        )

    return df_out


def build(df: pd.DataFrame, targets: Optional[List[str]] = None) -> pd.DataFrame:
    """Transform cleaned telemetry DataFrame into a model-ready feature matrix."""
    target_list = targets or ["NO3", "Turbidity"]
    df_matrix = df.copy()

    df_matrix = add_lag_features(df_matrix, SENSOR_COLS)
    df_matrix = add_rolling_features(df_matrix, SENSOR_COLS)
    df_matrix = add_cyclical_features(df_matrix)
    df_matrix = add_domain_features(df_matrix)

    df_matrix = df_matrix.drop(columns=["hour", "day_of_week", "month", "Timestamp"], errors="ignore")
    df_matrix = df_matrix.dropna(subset=target_list).reset_index(drop=True)
    df_matrix = df_matrix.replace([np.inf, -np.inf], np.nan)

    for col in df_matrix.columns:
        if df_matrix[col].isnull().any():
            median_val = df_matrix[col].median()
            df_matrix[col] = df_matrix[col].fillna(0.0 if np.isnan(median_val) else median_val)

    logger.info("Engineered feature matrix: %d features across %d rows", df_matrix.shape[1], len(df_matrix))
    return df_matrix


def get_feature_cols(df: pd.DataFrame, targets: List[str]) -> List[str]:
    """Extract predictor feature names excluding supervised targets."""
    target_set = set(targets)
    return [col for col in df.columns if col not in target_set]
