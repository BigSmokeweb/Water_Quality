import unittest
from llm.advisor import TelemetryAdvisoryEngine, generate_rule_based_advisory, format_context_payload


class TestAdvisoryEngine(unittest.TestCase):
    def test_rule_based_advisory_nominal(self):
        telemetry = {"NO3": 2.5, "Turbidity": 15.0, "Temp": 21.0}
        forecast = {
            "NO3": [2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
            "Turbidity": [16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
            "timestamps": ["2024-01-01T01:00:00"] * 6,
        }
        report = generate_rule_based_advisory(telemetry, forecast)
        self.assertIn("NORMAL", report)
        self.assertIn("baseline monitoring", report)

    def test_rule_based_advisory_excursion(self):
        telemetry = {"NO3": 12.5, "Turbidity": 120.0, "Temp": 26.0}
        forecast = {
            "NO3": [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            "Turbidity": [130.0, 140.0, 150.0, 160.0, 170.0, 180.0],
            "timestamps": ["2024-01-01T01:00:00"] * 6,
        }
        report = generate_rule_based_advisory(telemetry, forecast)
        self.assertIn("ALERT", report)
        self.assertIn("Suspend non-essential intake", report)

    def test_format_context_payload(self):
        telemetry = {"NO3": 5.0, "Turbidity": 20.0}
        forecast = {"NO3": [5.1, 5.2], "timestamps": ["2024-01-01T01:00:00", "2024-01-01T02:00:00"]}
        payload = format_context_payload(telemetry, forecast)
        self.assertIn("telemetry_readings", payload)
        self.assertIn("operational_thresholds", payload)


if __name__ == "__main__":
    unittest.main()
