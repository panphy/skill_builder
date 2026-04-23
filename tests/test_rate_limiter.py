import unittest

from rate_limiter import RATE_LIMITS_DDL, RATE_LIMITS_INDEX_DDL, _refill_tokens, _seconds_until_next_token


class TestRateLimiterMath(unittest.TestCase):
    def test_refill_tokens_accumulates_proportionally(self):
        self.assertEqual(_refill_tokens(0, 1800, 10, 3600), 5.0)

    def test_refill_tokens_caps_at_capacity(self):
        self.assertEqual(_refill_tokens(9, 1800, 10, 3600), 10.0)

    def test_seconds_until_next_token(self):
        self.assertEqual(_seconds_until_next_token(0, 10, 3600), 360)
        self.assertEqual(_seconds_until_next_token(0.5, 10, 3600), 180)
        self.assertEqual(_seconds_until_next_token(1, 10, 3600), 0)

    def test_rate_limit_indexes_are_applied_after_migration(self):
        self.assertNotIn("create index", RATE_LIMITS_DDL.lower())
        self.assertIn("last_refill_at", RATE_LIMITS_INDEX_DDL)


if __name__ == "__main__":
    unittest.main()
