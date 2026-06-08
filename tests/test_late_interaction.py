import unittest

import numpy as np

from turboragger.contracts import RetrievalResult
from turboragger.late_interaction import maxsim_score, rerank_candidates_with_maxsim


class LateInteractionTests(unittest.TestCase):
    def test_maxsim_score_averages_best_doc_token_matches(self):
        query_tokens = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc_tokens = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        score = maxsim_score(query_tokens, doc_tokens)

        self.assertAlmostEqual(score, 0.5)

    def test_rerank_candidates_with_maxsim_reorders_base_candidates(self):
        candidates = [
            RetrievalResult("d1", 0.9, "base"),
            RetrievalResult("d2", 0.8, "base"),
        ]
        query_tokens = np.array([[0.0, 1.0]], dtype=np.float32)
        doc_tokens = {
            "d1": np.array([[1.0, 0.0]], dtype=np.float32),
            "d2": np.array([[0.0, 1.0]], dtype=np.float32),
        }

        reranked = rerank_candidates_with_maxsim(
            candidates,
            query_tokens=query_tokens,
            doc_tokens_by_id=doc_tokens,
            source="late",
            limit=2,
        )

        self.assertEqual([result.doc_id for result in reranked], ["d2", "d1"])
        self.assertEqual(reranked[0].source, "late")
        self.assertGreater(reranked[0].score, reranked[1].score)


if __name__ == "__main__":
    unittest.main()
