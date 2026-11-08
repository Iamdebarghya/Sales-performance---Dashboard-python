import unittest
import pandas as pd

class SalesDataContractTests(unittest.TestCase):
    def test_region_aggregation(self):
        df = pd.DataFrame({"Region":["East","West","East"],"Sales":[10,20,5]})
        totals = df.groupby("Region")["Sales"].sum().to_dict()
        self.assertEqual(totals, {"East":15,"West":20})

    def test_numeric_sales_column(self):
        df = pd.DataFrame({"Sales":[100,200]})
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Sales"]))
        self.assertEqual(float(df["Sales"].sum()), 300.0)

    def test_date_parsing_handles_invalid_values(self):
        dates = pd.to_datetime(pd.Series(["2024-01-01","not-a-date"]), errors="coerce")
        self.assertEqual(int(dates.notna().sum()), 1)

if __name__ == "__main__":
    unittest.main()
