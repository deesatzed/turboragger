from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from turboragger.embedder_probe import candidate_cache_paths, find_model_snapshot, probe_candidate_path


class EmbedderProbeTests(unittest.TestCase):
    def test_find_model_snapshot_returns_complete_cached_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = (
                root
                / "models--BAAI--bge-m3"
                / "snapshots"
                / "abc123"
            )
            snapshot.mkdir(parents=True)
            for filename in ["config.json", "model.safetensors", "tokenizer.json"]:
                (snapshot / filename).write_text("{}")

            found = find_model_snapshot("BAAI/bge-m3", cache_roots=[root])

            self.assertEqual(found, snapshot)

    def test_probe_candidate_path_marks_stub_as_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "modules.json").write_text("{}")

            result = probe_candidate_path(path)

            self.assertFalse(result["complete"])
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("config.json", result["missing_files"])
            self.assertIn("model.safetensors or pytorch_model.bin", result["missing_files"])

    def test_candidate_cache_paths_uses_huggingface_repo_layout(self):
        root = Path("/tmp/hf")

        paths = candidate_cache_paths("Qwen/Qwen3-Embedding-0.6B", cache_roots=[root])

        self.assertEqual(
            paths,
            [root / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots"],
        )


if __name__ == "__main__":
    unittest.main()
