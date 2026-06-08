from __future__ import annotations

import unittest

from turboragger.contracts import RetrievalResult
from turboragger.score_fusion import ScoreFusionRetriever, minmax_normalize_results


class StaticRetriever:
    def __init__(self, results):
        self.results = results
        self.requested_k = []

    def retrieve(self, query: str, k: int = 100):
        self.requested_k.append(k)
        return self.results[:k]


class ScoreFusionTests(unittest.TestCase):
    def test_minmax_normalize_results_maps_scores_to_zero_one(self):
        normalized = minmax_normalize_results(
            [
                RetrievalResult("d1", 4.0, "a"),
                RetrievalResult("d2", 2.0, "a"),
                RetrievalResult("d3", 2.0, "a"),
            ]
        )

        self.assertEqual(normalized, {"d1": 1.0, "d2": 0.0, "d3": 0.0})

    def test_score_fusion_retriever_sums_normalized_branch_scores(self):
        retriever = ScoreFusionRetriever(
            {
                "a": StaticRetriever(
                    [
                        RetrievalResult("d1", 10.0, "a"),
                        RetrievalResult("d2", 5.0, "a"),
                    ]
                ),
                "b": StaticRetriever(
                    [
                        RetrievalResult("d2", 9.0, "b"),
                        RetrievalResult("d3", 1.0, "b"),
                    ]
                ),
            },
            source="fusion",
        )

        results = retriever.retrieve("query", k=3)

        self.assertEqual([result.doc_id for result in results], ["d1", "d2", "d3"])
        self.assertEqual(results[0].source, "fusion")
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, 1.0)

    def test_mnz_mode_boosts_documents_seen_by_multiple_branches(self):
        retriever = ScoreFusionRetriever(
            {
                "a": StaticRetriever(
                    [
                        RetrievalResult("d1", 10.0, "a"),
                        RetrievalResult("d2", 6.0, "a"),
                        RetrievalResult("d4", 2.0, "a"),
                    ]
                ),
                "b": StaticRetriever(
                    [
                        RetrievalResult("d2", 9.0, "b"),
                        RetrievalResult("d3", 1.0, "b"),
                    ]
                ),
            },
            source="fusion",
            mode="mnz",
        )

        results = retriever.retrieve("query", k=3)

        self.assertEqual([result.doc_id for result in results], ["d2", "d1", "d3"])
        self.assertAlmostEqual(results[0].score, 3.0)
        self.assertAlmostEqual(results[1].score, 1.0)

    def test_rank_score_mode_adds_reciprocal_rank_bonus(self):
        retriever = ScoreFusionRetriever(
            {
                "a": StaticRetriever(
                    [
                        RetrievalResult("high", 10.0, "a"),
                        RetrievalResult("early", 9.0, "a"),
                    ]
                ),
                "b": StaticRetriever(
                    [
                        RetrievalResult("early", 1.0, "b"),
                        RetrievalResult("high", 0.0, "b"),
                    ]
                ),
            },
            mode="rank_score",
            rank_weight=100.0,
            rrf_k=0,
        )

        results = retriever.retrieve("query", k=2)

        self.assertEqual(results[0].doc_id, "early")
        self.assertEqual(retriever.mode, "rank_score")

    def test_unknown_fusion_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ScoreFusionRetriever({"a": StaticRetriever([])}, mode="unknown")

    def test_branch_k_retrieves_deeper_branch_pool_than_final_output(self):
        branch_a = StaticRetriever(
            [
                RetrievalResult("d1", 10.0, "a"),
                RetrievalResult("d2", 9.0, "a"),
                RetrievalResult("d3", 8.5, "a"),
                RetrievalResult("d6", 2.0, "a"),
            ]
        )
        branch_b = StaticRetriever(
            [
                RetrievalResult("d4", 10.0, "b"),
                RetrievalResult("d3", 9.0, "b"),
                RetrievalResult("d5", 8.0, "b"),
                RetrievalResult("d7", 2.0, "b"),
            ]
        )
        retriever = ScoreFusionRetriever(
            {"a": branch_a, "b": branch_b},
            source="fusion",
            branch_k=4,
        )

        results = retriever.retrieve("query", k=2)

        self.assertEqual(branch_a.requested_k, [4])
        self.assertEqual(branch_b.requested_k, [4])
        self.assertEqual(len(results), 2)
        self.assertIn("d3", [result.doc_id for result in results])

    def test_branch_k_cannot_be_less_than_one(self):
        with self.assertRaises(ValueError):
            ScoreFusionRetriever({"a": StaticRetriever([])}, branch_k=0)


if __name__ == "__main__":
    unittest.main()
