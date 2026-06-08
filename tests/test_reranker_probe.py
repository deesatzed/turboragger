from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from turboragger.reranker_probe import find_reranker_snapshot, probe_reranker_path


class RerankerProbeTests(unittest.TestCase):
    def test_find_reranker_snapshot_returns_complete_cached_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "models--BAAI--bge-reranker-base" / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            for filename in ["config.json", "model.safetensors", "tokenizer.json"]:
                (snapshot / filename).write_text("{}")

            found = find_reranker_snapshot("BAAI/bge-reranker-base", cache_roots=[root])

            self.assertEqual(found, snapshot)

    def test_probe_reranker_path_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "config.json").write_text("{}")
            (path / "tokenizer.json").write_text("{}")

            result = probe_reranker_path(path)

            self.assertFalse(result["complete"])
            self.assertIn("model.safetensors or pytorch_model.bin", result["missing_files"])


if __name__ == "__main__":
    unittest.main()
