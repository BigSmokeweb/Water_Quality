import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENSOR_COLUMNS: List[str] = ["NO3", "Turbidity", "Temp", "Conductivity"]
IQR_MULTIPLIER: float = 3.0


def load_raw_dataset(filepath: Union[str, Path]) -> pd.DataFrame:
    """Ingest CSV telemetry and parse timestamp index."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry data source not found: {path}")

    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def impute_missing_telemetry(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    max_interpolation_gap_hours: int = 6,
) -> pd.DataFrame:
    """
    Impute missing telemetry observations.
    Uses time-based interpolation for short outages, followed by
    forward/backward propagation for sustained sensor dropouts.
    """
    cols = sensor_cols or SENSOR_COLUMNS
    df_out = df.copy()

    if "Timestamp" in df_out.columns:
        df_out = df_out.set_index("Timestamp")

    for col in cols:
        if col not in df_out.columns:
            continue
        df_out[col] = df_out[col].interpolate(
            method="time", limit=max_interpolation_gap_hours
        )
        df_out[col] = df_out[col].ffill().bfill()

    if "Timestamp" not in df_out.columns:
        df_out = df_out.reset_index()

    return df_out


def winsorize_sensor_anomalies(
    df: pd.DataFrame,
    sensor_cols: Optional[List[str]] = None,
    iqr_factor: float = IQR_MULTIPLIER,
) -> pd.DataFrame:
    """
    Apply statistical winsorization to suppress sensor spikes caused by fouling or transmission noise.
    """
    cols = sensor_cols or SENSOR_COLUMNS
    df_out = df.copy()
    anomaly_counts = {}

    for col in cols:
        if col not in df_out.columns:
            continue
        q25, q75 = df_out[col].quantile([0.25, 0.75])
        iqr = q75 - q25
        lower_bound = q25 - iqr_factor * iqr
        upper_bound = q75 + iqr_factor * iqr

        count = int(((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum())
        df_out[col] = df_out[col].clip(lower_bound, upper_bound)
        if count > 0:
            anomaly_counts[col] = count

    if anomaly_counts:
        logger.debug("Winsorized anomalies: %s", anomaly_counts)

    return df_out


def derive_temporal_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standard calendar and hourly indices from Timestamp."""
    df_out = df.copy()
    if "Timestamp" not in df_out.columns:
        raise ValueError("Timestamp column is required to derive temporal attributes.")

    df_out["hour"] = df_out["Timestamp"].dt.hour
    df_out["day_of_week"] = df_out["Timestamp"].dt.dayofweek
    df_out["month"] = df_out["Timestamp"].dt.month
    return df_out


def run(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Execute complete ingestion and cleaning pipeline."""
    from config import DATA_PATH

    target_path = Path(filepath or DATA_PATH)
    logger.info("Executing telemetry ingestion pipeline for %s", target_path)

    df = load_raw_dataset(target_path)
    df = df.drop(columns=["Dayofweek", "Month"], errors="ignore")
    df = impute_missing_telemetry(df)
    df = winsorize_sensor_anomalies(df)
    df = derive_temporal_attributes(df)

    logger.info("Pipeline complete. Output dataset shape: %s", df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    data = run()
    logger.info("Summary stats:\n%s", data.describe().to_string())
