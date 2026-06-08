from __future__ import annotations

import unittest
from pathlib import Path

from turboragger.local_model_inventory import classify_model_file, mark_benchmarked_entries, summarize_inventory


class LocalModelInventoryTests(unittest.TestCase):
    def test_classifies_goal_priority_embedding_model(self):
        result = classify_model_file(
            Path("/cache/models--BAAI--bge-m3/snapshots/abc/model.safetensors")
        )

        self.assertEqual(result["category"], "goal_priority_embedding")
        self.assertTrue(result["sota_relevant"])
        self.assertIn("bge-m3", result["signals"])

    def test_classifies_known_measured_local_models(self):
        result = classify_model_file(
            Path("/cache/Xenova/bge-small-en-v1.5/onnx/model_quantized.onnx")
        )

        self.assertEqual(result["category"], "known_measured_local_model")
        self.assertFalse(result["unmeasured_sota_candidate"])

    def test_qwen_language_model_is_not_embedding_candidate(self):
        result = classify_model_file(
            Path("/models/qwen2.5-0.5b-hyperspace-v1/model.safetensors")
        )

        self.assertEqual(result["category"], "general_language_model")
        self.assertFalse(result["sota_relevant"])

    def test_summary_tracks_unmeasured_sota_candidates(self):
        entries = [
            classify_model_file(Path("/cache/models--BAAI--bge-m3/snapshots/abc/model.safetensors")),
            classify_model_file(Path("/cache/Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx")),
        ]

        summary = summarize_inventory(entries)

        self.assertEqual(summary["file_count"], 2)
        self.assertEqual(summary["unmeasured_sota_candidate_count"], 1)
        self.assertEqual(summary["categories"]["goal_priority_embedding"], 1)

    def test_bce_embedding_is_unmeasured_embedding_candidate(self):
        result = classify_model_file(
            Path("/huggingface.co/maidalun1020/bce-embedding-base_v1/pytorch_model.bin")
        )

        self.assertEqual(result["category"], "goal_priority_embedding")
        self.assertTrue(result["unmeasured_sota_candidate"])
        self.assertIn("bce-embedding", result["signals"])

    def test_benchmarked_model_dir_clears_unmeasured_status(self):
        entry = classify_model_file(
            Path("/huggingface.co/maidalun1020/bce-embedding-base_v1/pytorch_model.bin")
        )

        marked = mark_benchmarked_entries(
            [entry],
            benchmarked_model_dirs={Path("/huggingface.co/maidalun1020/bce-embedding-base_v1")},
        )

        self.assertTrue(marked[0]["benchmarked"])
        self.assertFalse(marked[0]["unmeasured_sota_candidate"])


if __name__ == "__main__":
    unittest.main()
