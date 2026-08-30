import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ARTIFACTS_DIR, N_CV_FOLDS, RANDOM_STATE, TARGETS
from data.pipeline import run as run_pipeline
from features.engineer import build as build_features, get_feature_cols

logger = logging.getLogger(__name__)


def walk_forward_split(
    n_samples: int,
    n_folds: int = N_CV_FOLDS,
    min_train_fraction: float = 0.60,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate sequential expanding training and out-of-sample validation index splits.
    Guarantees strict temporal causality without future data leakage.
    """
    min_train_size = int(n_samples * min_train_fraction)
    val_fold_size = (n_samples - min_train_size) // n_folds

    for fold_idx in range(n_folds):
        train_end = min_train_size + fold_idx * val_fold_size
        val_end = train_end + val_fold_size
        yield np.arange(train_end), np.arange(train_end, val_end)


def construct_regressor(random_state: int = RANDOM_STATE) -> XGBRegressor:
    """Instantiate standard XGBoost regressor for tabular temporal regression."""
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def train_target_model(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Train and cross-validate regression model for a single water quality parameter."""
    feature_cols = get_feature_cols(df, TARGETS)
    X = df[feature_cols].values
    y = df[target].values

    logger.info("Training %s estimator on %d observations with %d features", target, len(X), len(feature_cols))

    fold_metrics: List[Dict[str, float]] = []
    for fold, (train_idx, val_idx) in enumerate(walk_forward_split(len(X), N_CV_FOLDS)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = construct_regressor()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        fold_metrics.append({"rmse": rmse, "mae": mae, "r2": r2})
        logger.info("Fold %d/%d [%s] — RMSE: %.4f, MAE: %.4f, R2: %.4f", fold + 1, N_CV_FOLDS, target, rmse, mae, r2)

    final_model = construct_regressor()
    final_model.fit(X, y, verbose=False)

    avg_metrics = {
        metric: float(np.mean([f[metric] for f in fold_metrics]))
        for metric in fold_metrics[0].keys()
    }
    logger.info("Mean CV Metrics [%s] — RMSE: %.4f, MAE: %.4f, R2: %.4f", target, avg_metrics["rmse"], avg_metrics["mae"], avg_metrics["r2"])

    sorted_importances = sorted(
        zip(feature_cols, [float(v) for v in final_model.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )[:15]

    return {
        "model": final_model,
        "feature_cols": feature_cols,
        "metrics": avg_metrics,
        "fold_metrics": fold_metrics,
        "importances": dict(sorted_importances),
    }


def _serialize_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _serialize_dict(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_serialize_dict(item) for item in data]
    if hasattr(data, "item"):
        return data.item()
    return data


def persist_artifacts(training_results: Dict[str, Any], output_directory: str = ARTIFACTS_DIR) -> None:
    """Serialize model binaries and evaluation metrics to disk."""
    out_path = Path(output_directory)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_report = {}

    for target, result in training_results.items():
        joblib.dump(result["model"], out_path / f"{target}_model.pkl")
        joblib.dump(result["feature_cols"], out_path / f"{target}_features.pkl")
        summary_report[target] = _serialize_dict({
            "metrics": result["metrics"],
            "importances": result["importances"],
        })
        logger.info("Serialized model artifact for %s", target)

    metrics_file = out_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info("Evaluation metrics report saved to %s", metrics_file)


def run() -> Dict[str, Any]:
    """Execute complete model training and validation workflow."""
    df_clean = run_pipeline()
    df_features = build_features(df_clean)

    results = {}
    for target in TARGETS:
        results[target] = train_target_model(df_features, target)

    persist_artifacts(results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
