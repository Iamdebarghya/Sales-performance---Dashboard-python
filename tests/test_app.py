import unittest
import pandas as pd
from app import validate_dataset, calculate_kpis, aggregate_by, monthly_trend

class DashboardAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.DataFrame({"Date":["2024-01-02","2024-01-19","2024-02-01","bad"],"Region":["East","West","East",None],"Sales":["10","20","5","bad"]})
    def test_empty_data_rejected(self):
        with self.assertRaises(ValueError): validate_dataset(pd.DataFrame())
    def test_kpis(self):
        self.assertEqual(calculate_kpis(self.df,"Sales"),{"total":35.0,"average":35.0/3,"valid_rows":3})
    def test_grouped(self):
        r=aggregate_by(self.df,"Region","Sales")
        self.assertEqual(r["category"].tolist(),["West","East"])
        self.assertEqual(r["value"].tolist(),[20,15])
    def test_monthly(self):
        r=monthly_trend(self.df,"Date","Sales")
        self.assertEqual(r["month"].tolist(),["2024-01","2024-02"])
        self.assertEqual(r["value"].tolist(),[30,5])

if __name__=="__main__": unittest.main()
