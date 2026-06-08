import unittest

from turboragger.metrics import ndcg_at_k, recall_at_k, score_ranked_results


class MetricTests(unittest.TestCase):
    def test_recall_at_k_counts_unique_relevant_docs_in_top_k(self):
        qrels = {"d1": 1, "d2": 1, "d3": 0}
        ranked = ["d4", "d1", "d1", "d2", "d3"]

        self.assertAlmostEqual(recall_at_k(ranked, qrels, k=4), 1.0)

    def test_recall_at_k_returns_zero_when_no_relevant_docs_exist(self):
        self.assertEqual(recall_at_k(["d1"], {}, k=100), 0.0)

    def test_ndcg_at_k_scores_relevant_ordering(self):
        qrels = {"d1": 2, "d2": 1, "d3": 0}

        self.assertAlmostEqual(ndcg_at_k(["d1", "d2", "d3"], qrels, k=3), 1.0)
        self.assertLess(ndcg_at_k(["d3", "d2", "d1"], qrels, k=3), 1.0)

    def test_score_ranked_results_reports_metric_surface(self):
        scores = score_ranked_results(
            {"q1": ["d1", "d2"], "q2": ["d4"]},
            {"q1": {"d2": 1}, "q2": {"d3": 1}},
            k_recall=2,
            k_ndcg=2,
        )

        self.assertEqual(scores["queries_tested"], 2)
        self.assertEqual(scores["failure_count"], 0)
        self.assertIn("Recall@100", scores["metrics"])
        self.assertIn("nDCG@10", scores["metrics"])


if __name__ == "__main__":
    unittest.main()
