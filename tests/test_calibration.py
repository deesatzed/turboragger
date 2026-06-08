from __future__ import annotations

import unittest

from turboragger.calibration import (
    calibrate_rank_score_fusion_parameters,
    calibrate_score_fusion_weights,
    fuse_score_results,
)
from turboragger.contracts import RetrievalResult


class CalibrationTests(unittest.TestCase):
    def test_fuse_score_results_applies_candidate_weights(self):
        fused = fuse_score_results(
            {
                "a": [
                    RetrievalResult("d1", 10.0, "a"),
                    RetrievalResult("d2", 0.0, "a"),
                ],
                "b": [
                    RetrievalResult("d2", 10.0, "b"),
                    RetrievalResult("d1", 0.0, "b"),
                ],
            },
            weights={"a": 0.1, "b": 1.0},
            limit=2,
        )

        self.assertEqual([result.doc_id for result in fused], ["d2", "d1"])
        self.assertAlmostEqual(fused[0].score, 1.0)
        self.assertAlmostEqual(fused[1].score, 0.1)

    def test_calibrate_score_fusion_weights_selects_best_dev_weight_set(self):
        calibration = calibrate_score_fusion_weights(
            {
                "q1": {
                    "a": [
                        RetrievalResult("d1", 10.0, "a"),
                        RetrievalResult("d2", 0.0, "a"),
                    ],
                    "b": [
                        RetrievalResult("d2", 10.0, "b"),
                        RetrievalResult("d1", 0.0, "b"),
                    ],
                }
            },
            {"q1": {"d2": 1}},
            weight_grid=[
                {"a": 1.0, "b": 0.0},
                {"a": 0.0, "b": 1.0},
            ],
            limit=2,
        )

        self.assertEqual(calibration["weights"], {"a": 0.0, "b": 1.0})
        self.assertAlmostEqual(calibration["metrics"]["nDCG@10"], 1.0)

    def test_calibrate_rank_score_fusion_parameters_selects_rank_weight(self):
        calibration = calibrate_rank_score_fusion_parameters(
            {
                "q1": {
                    "a": [
                        RetrievalResult("aaa_high", 10.0, "a"),
                        RetrievalResult("zzz_early", 9.0, "a"),
                    ],
                    "b": [
                        RetrievalResult("zzz_early", 1.0, "b"),
                    ],
                }
            },
            {"q1": {"zzz_early": 1}},
            weights={"a": 1.0, "b": 1.0},
            rank_weight_grid=[0.0, 100.0],
            rrf_k=0,
            limit=2,
        )

        self.assertEqual(calibration["rank_weight"], 100.0)
        self.assertAlmostEqual(calibration["metrics"]["nDCG@10"], 1.0)


if __name__ == "__main__":
    unittest.main()
