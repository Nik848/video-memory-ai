import unittest

from app.services.job_queue import _next_backoff_seconds


class JobQueueBackoffTests(unittest.TestCase):
    def test_backoff_increases(self):
        self.assertEqual(_next_backoff_seconds(1), 10)
        self.assertEqual(_next_backoff_seconds(2), 20)
        self.assertEqual(_next_backoff_seconds(3), 40)

    def test_backoff_caps(self):
        self.assertLessEqual(_next_backoff_seconds(20), 300)


if __name__ == "__main__":
    unittest.main()
