from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from turboragger.benchmark import build_candidate_payload, write_candidate_run, write_leaderboard


class BenchmarkArtifactTests(unittest.TestCase):
    def test_build_candidate_payload_records_required_metric_surface(self):
        payload = build_candidate_payload(
            run_id="bm25_20260608T132000Z",
            candidate_name="bm25",
            retrieval_mode="lexical",
            dataset={"status": "found", "fingerprint": {"dataset_sha256": "abc"}},
            retriever_config={"model": "rank_bm25.BM25Okapi", "top_k": 100},
            dependency={"rank_bm25": {"usable": True}},
            report={
                "runs": {"q1": ["d1", "d2"]},
                "branch_outputs": {"q1": {"bm25": [{"doc_id": "d1", "score": 2.0, "source": "bm25"}]}},
                "failures": {},
            },
            scores={
                "queries_tested": 1,
                "failure_count": 0,
                "metrics": {"nDCG@10": 1.0, "Recall@100": 0.5},
                "per_query_scores": {"ndcg@10": [1.0], "recall@100": [0.5]},
            },
            command="PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate bm25",
            hyperparameters={"k": 100},
            runtime_seconds=1.25,
        )

        self.assertEqual(payload["run_id"], "bm25_20260608T132000Z")
        self.assertEqual(payload["candidate"], "bm25")
        self.assertEqual(payload["retrieval_mode"], "lexical")
        self.assertEqual(payload["metrics"]["nDCG@10"], 1.0)
        self.assertEqual(payload["metrics"]["Recall@100"], 0.5)
        self.assertEqual(payload["query_count"], 1)
        self.assertEqual(payload["failure_count"], 0)
        self.assertEqual(payload["dataset"]["fingerprint"]["dataset_sha256"], "abc")
        self.assertIn("per_query_scores", payload)
        self.assertIn("runs", payload)
        self.assertIn("branch_outputs", payload)

    def test_write_leaderboard_sorts_saved_runs_by_ndcg_descending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low = {
                "run_id": "low",
                "candidate": "bm25",
                "retrieval_mode": "lexical",
                "metrics": {"nDCG@10": 0.2, "Recall@100": 0.3},
                "query_count": 1,
                "failure_count": 0,
                "artifact_path": "artifacts/runs/low.json",
            }
            high = {
                "run_id": "high",
                "candidate": "dense",
                "retrieval_mode": "dense",
                "metrics": {"nDCG@10": 0.4, "Recall@100": 0.35},
                "query_count": 1,
                "failure_count": 0,
                "artifact_path": "artifacts/runs/high.json",
            }

            write_candidate_run(root, low)
            write_candidate_run(root, high)
            leaderboard_path = write_leaderboard(root)

            leaderboard = json.loads(leaderboard_path.read_text())
            self.assertEqual([entry["run_id"] for entry in leaderboard["runs"]], ["high", "low"])
            self.assertEqual(leaderboard["best_run"]["run_id"], "high")
            self.assertEqual(leaderboard["sort_key"], "nDCG@10 desc")


if __name__ == "__main__":
    unittest.main()
