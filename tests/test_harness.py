import unittest

from turboragger.contracts import RetrievalResult
from turboragger.harness import RetrievalHarness


class StaticRetriever:
    def retrieve(self, query: str, k: int):
        return [
            RetrievalResult(doc_id=f"{query}-a", score=0.9, source="dense"),
            RetrievalResult(doc_id=f"{query}-b", score=0.4, source="dense"),
        ][:k]


class HarnessTests(unittest.TestCase):
    def test_dense_only_harness_returns_ranked_doc_ids_by_query_id(self):
        harness = RetrievalHarness({"dense": StaticRetriever()})

        runs = harness.run_queries({"q1": "query-one"}, k=10)

        self.assertEqual(runs, {"q1": ["query-one-a", "query-one-b"]})

    def test_dense_only_harness_records_branch_outputs(self):
        harness = RetrievalHarness({"dense": StaticRetriever()})

        report = harness.run_with_report({"q1": "query-one"}, k=1)

        self.assertEqual(report["runs"]["q1"], ["query-one-a"])
        self.assertEqual(report["branch_outputs"]["q1"]["dense"][0]["doc_id"], "query-one-a")


if __name__ == "__main__":
    unittest.main()
