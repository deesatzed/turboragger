from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from turboragger.leann_bridge import LeannMiniLMRetriever


class FakeSearchResult:
    def __init__(self, doc_id: str, score: float):
        self.metadata = {"node_id": doc_id}
        self.score = score


class LeannBridgeTests(unittest.TestCase):
    def test_leann_minilm_retriever_builds_no_recompute_index_and_maps_results(self):
        builder = mock.Mock()
        searcher = mock.Mock()
        searcher.search.return_value = [FakeSearchResult("d1", 0.9)]

        with tempfile.TemporaryDirectory() as temp_dir:
            retriever = LeannMiniLMRetriever(
                {"d1": {"title": "Aspirin", "text": "Blood clotting"}},
                model_path=Path(temp_dir),
                index_path=Path(temp_dir) / "idx",
                builder_factory=mock.Mock(return_value=builder),
                searcher_factory=mock.Mock(return_value=searcher),
            )

            results = retriever.retrieve("clotting", k=1)

        builder.add_text.assert_called_once()
        builder.build_index.assert_called_once()
        self.assertEqual(searcher.search.call_args.kwargs["top_k"], 1)
        self.assertEqual(results[0].doc_id, "d1")
        self.assertEqual(results[0].source, "leann_minilm_no_recompute")


if __name__ == "__main__":
    unittest.main()
