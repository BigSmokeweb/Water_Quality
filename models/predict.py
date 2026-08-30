import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ARTIFACTS_DIR, FORECAST_STEPS, TARGETS
from features.engineer import (
    SENSOR_COLS,
    add_cyclical_features,
    add_domain_features,
    add_lag_features,
    add_rolling_features,
)

logger = logging.getLogger(__name__)


def load_models(artifacts_dir: Union[str, Path] = ARTIFACTS_DIR) -> Dict[str, Dict[str, Any]]:
    """Load serialized model regressors and corresponding feature schemas."""
    dir_path = Path(artifacts_dir)
    models = {}

    for target in TARGETS:
        model_file = dir_path / f"{target}_model.pkl"
        features_file = dir_path / f"{target}_features.pkl"

        if not model_file.exists() or not features_file.exists():
            raise FileNotFoundError(
                f"Missing model artifact for {target} at {dir_path}. Ensure models are trained first."
            )

        models[target] = {
            "model": joblib.load(model_file),
            "feature_cols": joblib.load(features_file),
        }

    return models


def forecast(
    df_history: pd.DataFrame,
    models: Dict[str, Dict[str, Any]],
    steps: int = FORECAST_STEPS,
) -> Dict[str, Any]:
    """
    Generate recursive autoregressive forecasts over a forward-looking horizon.
    """
    if "Timestamp" not in df_history.columns:
        raise ValueError("Historical dataframe must contain a 'Timestamp' column.")

    last_ts = df_history["Timestamp"].iloc[-1]
    future_timestamps = pd.date_range(
        last_ts + pd.Timedelta(hours=1), periods=steps, freq="1h"
    )

    predictions: Dict[str, List[float]] = {t: [] for t in TARGETS}
    rolling_buffer = df_history.copy()

    for step_idx in range(steps):
        buf = rolling_buffer.copy()
        buf["hour"] = buf["Timestamp"].dt.hour
        buf["day_of_week"] = buf["Timestamp"].dt.dayofweek
        buf["month"] = buf["Timestamp"].dt.month

        buf = add_lag_features(buf, SENSOR_COLS)
        buf = add_rolling_features(buf, SENSOR_COLS)
        buf = add_cyclical_features(buf)
        buf = add_domain_features(buf)
        buf = buf.drop(columns=["hour", "day_of_week", "month"], errors="ignore")
        buf = buf.dropna()

        latest_record = buf.iloc[[-1]]
        step_outputs: Dict[str, float] = {}

        for target, model_meta in models.items():
            expected_features = model_meta["feature_cols"]
            available_cols = [c for c in expected_features if c in latest_record.columns]
            missing_cols = [c for c in expected_features if c not in latest_record.columns]

            feature_vector = latest_record[available_cols].copy()
            for mc in missing_cols:
                feature_vector[mc] = 0.0

            feature_vector = feature_vector[expected_features]
            predicted_val = float(model_meta["model"].predict(feature_vector.values)[0])
            bounded_val = max(0.0, round(predicted_val, 4))
            predictions[target].append(bounded_val)
            step_outputs[target] = bounded_val

        next_observation = rolling_buffer.iloc[[-1]].copy()
        next_observation["Timestamp"] = future_timestamps[step_idx]
        for target, val in step_outputs.items():
            next_observation[target] = val

        rolling_buffer = pd.concat([rolling_buffer, next_observation], ignore_index=True)

    result = {
        **predictions,
        "timestamps": [ts.isoformat() for ts in future_timestamps],
    }
    return result


if __name__ == "__main__":
    from data.pipeline import run as run_pipeline
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df_cleaned = run_pipeline()
    loaded_models = load_models()
    horizon_forecasts = forecast(df_cleaned, loaded_models, steps=6)
    logger.info("Generated 6-hour forecast:\n%s", json.dumps(horizon_forecasts, indent=2))
