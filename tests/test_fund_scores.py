import unittest

import pandas as pd

from modules.fund_scores import merge_score_frames


class FundScoreTests(unittest.TestCase):
    def test_merge_score_frames_does_not_cartesian_join_coarse_keys(self):
        returns = pd.DataFrame(
            [
                {"fund_name": "Canara Robeco Large Cap Fund (G)", "coarse_fund_key": "canara robeco", "returns_5y_score": 60.0},
                {"fund_name": "Canara Robeco Equity Fund (G)", "coarse_fund_key": "canara robeco", "returns_5y_score": 40.0},
            ]
        )
        risk = pd.DataFrame(
            [
                {"fund_name": "Canara Robeco Large Cap Fund (G)", "coarse_fund_key": "canara robeco", "risk_metrics_score": 70.0},
                {"fund_name": "Canara Robeco Equity Fund (G)", "coarse_fund_key": "canara robeco", "risk_metrics_score": 30.0},
            ]
        )

        merged = merge_score_frames([returns, risk])

        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged["fund_name"]), {"Canara Robeco Large Cap Fund (G)", "Canara Robeco Equity Fund (G)"})


if __name__ == "__main__":
    unittest.main()