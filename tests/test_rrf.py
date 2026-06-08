import unittest

from turboragger.rrf import reciprocal_rank_fusion


class RRFTests(unittest.TestCase):
    def test_rrf_prefers_documents_appearing_in_multiple_ranked_lists(self):
        fused = reciprocal_rank_fusion(
            {
                "dense": [("a", 0.9), ("shared", 0.4), ("dense-only", 0.3)],
                "sparse": [("shared", 11.0), ("b", 9.0)],
            },
            k=60,
        )

        self.assertEqual(fused[0].doc_id, "shared")
        self.assertGreater(fused[0].score, fused[1].score)

    def test_rrf_limits_results(self):
        fused = reciprocal_rank_fusion(
            {
                "dense": [("a", 1.0), ("b", 0.5)],
                "sparse": [("c", 1.0), ("d", 0.5)],
            },
            limit=2,
        )

        self.assertEqual(len(fused), 2)


if __name__ == "__main__":
    unittest.main()
