import unittest

import pandas as pd

from modules.overweight_underweight import compute_overweight_score


class OverweightUnderweightTests(unittest.TestCase):
    def test_1m_uses_the_two_disclosures_before_the_return_month(self):
        fund_weights = pd.DataFrame(
            [
                {"fund_name": "Fund A", "date": "2026-03-31", "sector": "Technology", "weight_pct": 1.0},
                {"fund_name": "Fund A", "date": "2026-04-30", "sector": "Technology", "weight_pct": 3.0},
                {"fund_name": "Fund A", "date": "2026-05-31", "sector": "Technology", "weight_pct": 8.0},
            ]
        )
        fund_weights["date"] = pd.to_datetime(fund_weights["date"])

        sector_returns = pd.DataFrame(
            [
                {
                    "Lookback": "1M",
                    "Stock": "Nifty 50",
                    "Return": 10.0,
                    "TargetDateUsed": "2026-06-01",
                    "AnchorDateUsed": "2026-07-01",
                }
            ]
        )

        sector_mapping = {"Nifty 50": ["Technology"]}

        price_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-01", "2026-07-01"]),
                "Nifty 50": [100.0, 110.0],
            }
        )

        result = compute_overweight_score(
            fund_weights=fund_weights,
            sector_returns=sector_returns,
            sector_mapping=sector_mapping,
            price_history=price_history,
            lookback_label="1M",
        )

        self.assertEqual(result["summary"]["signal_start_date"], pd.Timestamp("2026-04-30"))
        self.assertEqual(result["summary"]["signal_end_date"], pd.Timestamp("2026-05-31"))

    def test_3m_uses_the_dominant_late_month_signal_when_returns_cluster_later(self):
        fund_weights = pd.DataFrame(
            [
                {"fund_name": "Fund A", "date": "2026-02-28", "sector": "Technology", "weight_pct": 1.0},
                {"fund_name": "Fund A", "date": "2026-03-31", "sector": "Technology", "weight_pct": 1.0},
                {"fund_name": "Fund A", "date": "2026-04-30", "sector": "Technology", "weight_pct": 3.0},
                {"fund_name": "Fund A", "date": "2026-05-31", "sector": "Technology", "weight_pct": 3.0},
                {"fund_name": "Fund A", "date": "2026-06-30", "sector": "Technology", "weight_pct": 8.0},
            ]
        )
        fund_weights["date"] = pd.to_datetime(fund_weights["date"])

        sector_returns = pd.DataFrame(
            [
                {
                    "Lookback": "3M",
                    "Stock": "Nifty 50",
                    "Return": 10.0,
                    "TargetDateUsed": "2026-04-01",
                    "AnchorDateUsed": "2026-07-01",
                }
            ]
        )

        sector_mapping = {"Nifty 50": ["Technology"]}

        price_history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01"]),
                "Nifty 50": [100.0, 100.0, 100.0, 110.0],
            }
        )

        result = compute_overweight_score(
            fund_weights=fund_weights,
            sector_returns=sector_returns,
            sector_mapping=sector_mapping,
            price_history=price_history,
            lookback_label="3M",
        )

        self.assertEqual(result["summary"]["signal_start_date"], pd.Timestamp("2026-04-30"))
        self.assertEqual(result["summary"]["signal_end_date"], pd.Timestamp("2026-05-31"))


if __name__ == "__main__":
    unittest.main()
