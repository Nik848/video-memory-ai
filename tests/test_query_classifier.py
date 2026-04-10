import unittest

from app.services.query_classifier import classify_query


class QueryClassifierTests(unittest.TestCase):
    def test_cluster_intent(self):
        self.assertEqual(
            classify_query("show me cluster themes")["intent"],
            "cluster_exploration",
        )

    def test_comparison_intent(self):
        self.assertEqual(
            classify_query("compare video A vs video B")["intent"],
            "comparison",
        )

    def test_fallback_intent(self):
        self.assertEqual(classify_query("hello there")["intent"], "general_search")


if __name__ == "__main__":
    unittest.main()
