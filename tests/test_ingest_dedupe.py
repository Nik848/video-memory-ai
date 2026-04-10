import unittest

from app.routes.ingest import _source_fingerprint


class IngestDedupeTests(unittest.TestCase):
    def test_fingerprint_same_for_url_case_changes(self):
        a = _source_fingerprint("HTTPS://example.com/Path", "u1")
        b = _source_fingerprint("https://example.com/path", "u1")
        self.assertEqual(a, b)

    def test_fingerprint_different_for_different_users(self):
        a = _source_fingerprint("https://example.com", "u1")
        b = _source_fingerprint("https://example.com", "u2")
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
