import unittest
import pandas as pd
from analytics import validate_dataset, calculate_kpis, aggregate_by, monthly_trend


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "Date": ["2024-01-02", "2024-01-19", "2024-02-01", "bad-date"],
            "Region": ["East", "West", "East", None],
            "Sales": ["10", "20", "5", "not-a-number"],
        })

    def test_validate_rejects_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_dataset(pd.DataFrame())

    def test_kpis_ignore_invalid_numeric_values(self):
        self.assertEqual(calculate_kpis(self.data, "Sales"), {
            "total": 35.0, "average": 35.0 / 3, "valid_rows": 3
        })

    def test_grouped_totals_are_descending(self):
        result = aggregate_by(self.data, "Region", "Sales")
        self.assertEqual(result["category"].tolist(), ["West", "East"])
        self.assertEqual(result["value"].tolist(), [20, 15])

    def test_monthly_trend_ignores_invalid_date_and_measure(self):
        result = monthly_trend(self.data, "Date", "Sales")
        self.assertEqual(result["month"].tolist(), ["2024-01", "2024-02"])
        self.assertEqual(result["value"].tolist(), [30, 5])

    def test_unknown_measure_is_reported(self):
        with self.assertRaisesRegex(KeyError, "Unknown measure"):
            calculate_kpis(self.data, "Profit")


if __name__ == "__main__":
    unittest.main()
