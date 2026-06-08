from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from turboragger.historical_audit import audit_newragcity_historical_claim


class HistoricalAuditTests(unittest.TestCase):
    def test_rejects_saved_claim_without_runs_and_with_invalid_metric_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "result.json"
            source_path = root / "benchmark.py"
            result_path.write_text(
                json.dumps(
                    {
                        "queries_tested": 2,
                        "metrics": {"nDCG@10": 0.51, "Recall@100": 1.5},
                        "per_query_scores": {
                            "ndcg@10": [1.0, 0.0],
                            "recall@100": [2.0, 1.0],
                        },
                    }
                )
            )
            source_path.write_text(
                "\n".join(
                    [
                        "def calculate_ndcg_at_k(relevances, k=10):",
                        "    ideal_relevances = sorted(relevances, reverse=True)",
                        "    return 1.0",
                        "for result in retrieved_results[:10]:",
                        "    retrieved_relevant_count += 1",
                        "for result in retrieved_results[:100]:",
                        "    retrieved_relevant_count += 1",
                    ]
                )
            )

            audit = audit_newragcity_historical_claim(result_path, source_path)

        self.assertEqual(audit["verdict"], "invalid")
        self.assertEqual(audit["claimed_metrics"]["nDCG@10"], 0.51)
        self.assertEqual(audit["recall_gt_1_count"], 1)
        self.assertFalse(audit["has_retrieved_doc_ids"])
        self.assertIn("top10_only_idcg", audit["issues"])
        self.assertIn("recall_double_count_risk", audit["issues"])


if __name__ == "__main__":
    unittest.main()
