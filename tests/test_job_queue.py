import unittest

from app.services.job_queue import _next_backoff_seconds
from app.config import JOB_BASE_BACKOFF_SECONDS, JOB_MAX_BACKOFF_SECONDS


class JobQueueBackoffTests(unittest.TestCase):
    def test_backoff_increases(self):
        self.assertEqual(_next_backoff_seconds(1), JOB_BASE_BACKOFF_SECONDS)
        self.assertEqual(_next_backoff_seconds(2), min(JOB_BASE_BACKOFF_SECONDS * 2, JOB_MAX_BACKOFF_SECONDS))
        self.assertEqual(_next_backoff_seconds(3), min(JOB_BASE_BACKOFF_SECONDS * 4, JOB_MAX_BACKOFF_SECONDS))

    def test_backoff_caps(self):
        self.assertLessEqual(_next_backoff_seconds(20), JOB_MAX_BACKOFF_SECONDS)


if __name__ == "__main__":
    unittest.main()
