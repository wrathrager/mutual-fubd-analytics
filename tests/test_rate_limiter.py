import unittest

from modules.moneycontrol_client import should_throttle


class RateLimiterTests(unittest.TestCase):
    def test_should_throttle_when_interval_has_not_elapsed(self):
        self.assertTrue(should_throttle(last_request_at=100.0, now=105.0, min_interval=10.0))

    def test_should_not_throttle_after_interval_has_elapsed(self):
        self.assertFalse(should_throttle(last_request_at=100.0, now=111.0, min_interval=10.0))

    def test_should_not_throttle_without_previous_request(self):
        self.assertFalse(should_throttle(last_request_at=None, now=10.0, min_interval=10.0))


if __name__ == "__main__":
    unittest.main()
