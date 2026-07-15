import unittest
from unittest.mock import patch

import pandas as pd

from app import build_comparative_snapshot


class ComparativeSnapshotTests(unittest.TestCase):
    def test_build_comparative_snapshot_handles_fetch_errors(self):
        fund_master = pd.DataFrame([{"fund_name": "Fund A", "isin": "IN123"}])

        with patch("app.fetch_nav_bundle", side_effect=Exception("boom")):
            snapshot = build_comparative_snapshot(fund_master)

        self.assertIsInstance(snapshot, pd.DataFrame)
        self.assertIn("number_of_holdings", snapshot.columns)
        self.assertIn("top_10_weight_pct", snapshot.columns)


if __name__ == "__main__":
    unittest.main()
