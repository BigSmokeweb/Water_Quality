import unittest
import numpy as np
import pandas as pd
from data.pipeline import (
    impute_missing_telemetry,
    winsorize_sensor_anomalies,
    derive_temporal_attributes,
)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        timestamps = pd.date_range("2024-01-01 00:00:00", periods=24, freq="1h")
        self.df = pd.DataFrame({
            "Timestamp": timestamps,
            "NO3": [1.0, 1.2, np.nan, np.nan, 1.5, 2.0] * 4,
            "Turbidity": [10.0, 12.0, 14.0, 500.0, 11.0, 13.0] * 4,
            "Temp": [22.0, 22.5, 23.0, 23.5, 24.0, 24.5] * 4,
            "Conductivity": [150.0, 155.0, 160.0, 165.0, 170.0, 175.0] * 4,
        })

    def test_impute_missing_telemetry(self):
        imputed = impute_missing_telemetry(self.df)
        self.assertEqual(imputed["NO3"].isnull().sum(), 0)
        self.assertEqual(len(imputed), len(self.df))

    def test_winsorize_sensor_anomalies(self):
        cleaned = winsorize_sensor_anomalies(self.df)
        self.assertLess(cleaned["Turbidity"].max(), 500.0)

    def test_derive_temporal_attributes(self):
        derived = derive_temporal_attributes(self.df)
        self.assertIn("hour", derived.columns)
        self.assertIn("day_of_week", derived.columns)
        self.assertIn("month", derived.columns)
        self.assertEqual(derived["hour"].iloc[0], 0)
        self.assertEqual(derived["hour"].iloc[5], 5)


if __name__ == "__main__":
    unittest.main()
