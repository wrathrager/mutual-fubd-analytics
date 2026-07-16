import unittest
from unittest.mock import patch

import streamlit as st

from modules.moneycontrol_client import fetch_nifty100_series


class StressBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_data.clear()

    @patch("modules.moneycontrol_client.fetch_json", side_effect=Exception("network blocked"))
    def test_fetch_nifty100_series_falls_back_to_local_prices(self, _mock_fetch_json):
        frame = fetch_nifty100_series()

        self.assertFalse(frame.empty)
        self.assertTrue({"date", "value"}.issubset(set(frame.columns)))
        self.assertTrue(frame["value"].notna().any())


if __name__ == "__main__":
    unittest.main()
