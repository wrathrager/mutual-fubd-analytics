import unittest
from unittest.mock import patch

import pandas as pd
import requests

from modules.moneycontrol_client import fetch_json, fetch_nifty100_series


class MoneycontrolClientTests(unittest.TestCase):
    @patch("modules.moneycontrol_client.requests.get")
    def test_fetch_nifty100_series_returns_empty_frame_on_http_error(self, mock_get):
        mock_get.side_effect = requests.HTTPError("403 forbidden")
        fetch_nifty100_series.clear()

        result = fetch_nifty100_series()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["date", "value"])


if __name__ == "__main__":
    unittest.main()
