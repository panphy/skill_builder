import unittest

from rate_limiter import _refill_tokens, _seconds_until_next_token


class TestRateLimiterMath(unittest.TestCase):
    def test_refill_tokens_accumulates_proportionally(self):
        self.assertEqual(_refill_tokens(0, 1800, 10, 3600), 5.0)

    def test_refill_tokens_caps_at_capacity(self):
        self.assertEqual(_refill_tokens(9, 1800, 10, 3600), 10.0)

    def test_seconds_until_next_token(self):
        self.assertEqual(_seconds_until_next_token(0, 10, 3600), 360)
        self.assertEqual(_seconds_until_next_token(0.5, 10, 3600), 180)
        self.assertEqual(_seconds_until_next_token(1, 10, 3600), 0)


if __name__ == "__main__":
    unittest.main()
