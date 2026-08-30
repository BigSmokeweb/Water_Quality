import unittest
import numpy as np
import pandas as pd
from features.engineer import (
    add_lag_features,
    add_rolling_features,
    add_cyclical_features,
    add_domain_features,
    build,
    get_feature_cols,
)


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2024-01-01 00:00:00", periods=50, freq="1h")
        self.df = pd.DataFrame({
            "Timestamp": timestamps,
            "NO3": np.linspace(1.0, 5.0, 50),
            "Turbidity": np.linspace(10.0, 30.0, 50),
            "Temp": np.linspace(20.0, 25.0, 50),
            "Conductivity": np.linspace(100.0, 200.0, 50),
            "hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek,
            "month": timestamps.month,
        })

    def test_add_lag_features(self):
        lagged = add_lag_features(self.df)
        self.assertIn("NO3_lag_1h", lagged.columns)
        self.assertIn("Turbidity_lag_6h", lagged.columns)

    def test_add_rolling_features(self):
        rolled = add_rolling_features(self.df)
        self.assertIn("NO3_roll_mean_3h", rolled.columns)
        self.assertIn("Turbidity_roll_std_6h", rolled.columns)

    def test_add_cyclical_features(self):
        cyc = add_cyclical_features(self.df)
        self.assertIn("hour_sin", cyc.columns)
        self.assertIn("hour_cos", cyc.columns)
        self.assertTrue(np.all(cyc["hour_sin"] >= -1.0) and np.all(cyc["hour_sin"] <= 1.0))

    def test_add_domain_features(self):
        dom = add_domain_features(self.df)
        self.assertIn("turb_temp_stress", dom.columns)
        self.assertIn("no3_cond_ratio", dom.columns)

    def test_build_feature_matrix(self):
        matrix = build(self.df, targets=["NO3", "Turbidity"])
        self.assertEqual(matrix.isnull().sum().sum(), 0)
        self.assertNotIn("Timestamp", matrix.columns)
        feature_cols = get_feature_cols(matrix, targets=["NO3", "Turbidity"])
        self.assertNotIn("NO3", feature_cols)
        self.assertNotIn("Turbidity", feature_cols)


if __name__ == "__main__":
    unittest.main()
