import json
import tempfile
import unittest
from pathlib import Path

from turboragger.artifacts import write_json_artifact
from turboragger.contracts import RetrievalResult, validate_ranked_results


class ContractAndArtifactTests(unittest.TestCase):
    def test_validate_ranked_results_accepts_retrieval_results(self):
        results = validate_ranked_results(
            [RetrievalResult(doc_id="d1", score=0.9), {"doc_id": "d2", "score": 0.4}]
        )

        self.assertEqual([result.doc_id for result in results], ["d1", "d2"])

    def test_validate_ranked_results_rejects_missing_doc_id(self):
        with self.assertRaises(ValueError):
            validate_ranked_results([{"score": 0.4}])

    def test_write_json_artifact_adds_timestamp_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json_artifact(
                Path(tmp) / "probe.json",
                {"status": "ok"},
                command="python3 scripts/probe_environment.py",
            )
            payload = json.loads(path.read_text())

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["command"], "python3 scripts/probe_environment.py")
        self.assertIn("timestamp_utc", payload)


if __name__ == "__main__":
    unittest.main()
