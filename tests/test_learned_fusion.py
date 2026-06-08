from __future__ import annotations

import unittest

from turboragger.contracts import RetrievalResult
from turboragger.learned_fusion import (
    CascadeFusionRetriever,
    LinearFeatureFusionRetriever,
    ModelFeatureFusionRetriever,
    build_graded_feature_rows,
    build_labeled_feature_rows,
    cascade_ranked_results,
)


class StaticRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query: str, k: int = 100):
        return self.results[:k]


class ProbabilityModel:
    def predict_proba(self, rows):
        return [[1.0 - row[-1], row[-1]] for row in rows]


class LearnedFusionTests(unittest.TestCase):
    def test_cascade_ranked_results_preserves_anchor_then_backfills(self):
        cascaded = cascade_ranked_results(
            [
                RetrievalResult("p1", 10.0, "primary"),
                RetrievalResult("shared", 9.0, "primary"),
                RetrievalResult("p3", 8.0, "primary"),
            ],
            [
                RetrievalResult("s1", 7.0, "secondary"),
                RetrievalResult("shared", 6.0, "secondary"),
                RetrievalResult("s3", 5.0, "secondary"),
            ],
            anchor_k=2,
            limit=4,
            source="cascade",
        )

        self.assertEqual([result.doc_id for result in cascaded], ["p1", "shared", "s1", "s3"])
        self.assertEqual(cascaded[0].source, "cascade")

    def test_cascade_fusion_retriever_uses_primary_anchor_and_secondary_fill(self):
        retriever = CascadeFusionRetriever(
            primary=StaticRetriever(
                [
                    RetrievalResult("p1", 10.0, "primary"),
                    RetrievalResult("p2", 9.0, "primary"),
                ]
            ),
            secondary=StaticRetriever(
                [
                    RetrievalResult("s1", 8.0, "secondary"),
                    RetrievalResult("p2", 7.0, "secondary"),
                ]
            ),
            anchor_k=1,
        )

        results = retriever.retrieve("query", k=3)

        self.assertEqual([result.doc_id for result in results], ["p1", "s1", "p2"])

    def test_linear_feature_fusion_combines_score_and_rank_weights(self):
        retriever = LinearFeatureFusionRetriever(
            {
                "a": StaticRetriever(
                    [
                        RetrievalResult("high_score", 10.0, "a"),
                        RetrievalResult("early_rank", 9.0, "a"),
                    ]
                ),
                "b": StaticRetriever(
                    [
                        RetrievalResult("early_rank", 1.0, "b"),
                        RetrievalResult("high_score", 0.0, "b"),
                    ]
                ),
            },
            score_weights={"a": 1.0},
            rank_weights={"b": 100.0},
            rrf_k=0,
        )

        results = retriever.retrieve("query", k=2)

        self.assertEqual([result.doc_id for result in results], ["early_rank", "high_score"])
        self.assertEqual(results[0].source, "linear_feature_fusion")

    def test_model_feature_fusion_scores_rows_with_predict_proba(self):
        retriever = ModelFeatureFusionRetriever(
            {
                "a": StaticRetriever(
                    [
                        RetrievalResult("single_branch", 10.0, "a"),
                        RetrievalResult("two_branches", 9.0, "a"),
                    ]
                ),
                "b": StaticRetriever(
                    [
                        RetrievalResult("two_branches", 1.0, "b"),
                    ]
                ),
            },
            branch_names=["a", "b"],
            model=ProbabilityModel(),
        )

        results = retriever.retrieve("query", k=2)

        self.assertEqual([result.doc_id for result in results], ["two_branches", "single_branch"])
        self.assertEqual(results[0].source, "model_feature_fusion")

    def test_build_labeled_feature_rows_marks_relevant_candidates(self):
        rows, labels, feature_names = build_labeled_feature_rows(
            {
                "q1": {
                    "a": [
                        RetrievalResult("a_relevant", 10.0, "a"),
                        RetrievalResult("z_negative", 0.0, "a"),
                    ],
                    "b": [
                        RetrievalResult("z_negative", 4.0, "b"),
                    ],
                }
            },
            {"q1": {"a_relevant": 2}},
            branch_names=["a", "b"],
            rrf_k=60,
        )

        self.assertEqual(labels, [1, 0])
        self.assertEqual(feature_names, ["score:a", "score:b", "rank:a", "rank:b", "presence_count"])
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0][0], rows[1][0])

    def test_build_graded_feature_rows_preserves_relevance_grades(self):
        rows, targets, feature_names = build_graded_feature_rows(
            {
                "q1": {
                    "a": [
                        RetrievalResult("high_grade", 10.0, "a"),
                        RetrievalResult("low_grade", 5.0, "a"),
                        RetrievalResult("negative", 1.0, "a"),
                    ],
                }
            },
            {"q1": {"high_grade": 2, "low_grade": 1}},
            branch_names=["a"],
            rrf_k=60,
        )

        self.assertEqual(targets, [2.0, 1.0, 0.0])
        self.assertEqual(feature_names, ["score:a", "rank:a", "presence_count"])
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
