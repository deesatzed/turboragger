import unittest

import numpy as np
import torch

from turboragger.dense import (
    DenseVectorIndex,
    dense_prf_query_vector,
    find_cached_minilm_snapshot,
    pad_token_batch,
    pool_last_hidden_state,
    pool_torch_last_hidden_state,
)


class DenseUtilityTests(unittest.TestCase):
    def test_dense_vector_index_returns_highest_cosine_match(self):
        index = DenseVectorIndex(
            doc_ids=["d1", "d2"],
            vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            source="dense-test",
        )

        results = index.search(np.array([0.9, 0.1], dtype=np.float32), k=2)

        self.assertEqual(results[0].doc_id, "d1")
        self.assertGreater(results[0].score, results[1].score)

    def test_find_cached_minilm_snapshot_returns_none_for_missing_root(self):
        self.assertIsNone(find_cached_minilm_snapshot(cache_root="/path/that/does/not/exist", include_default=False))

    def test_dense_prf_query_vector_adds_feedback_centroid(self):
        expanded = dense_prf_query_vector(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
            query_weight=1.0,
            feedback_weight=2.0,
        )

        expected = np.array([1.0, 2.0], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        self.assertTrue(np.allclose(expanded, expected))

    def test_pad_token_batch_pads_ids_masks_and_types(self):
        batch = pad_token_batch(
            input_ids=[[101, 102], [101]],
            attention_masks=[[1, 1], [1]],
            token_type_ids=[[0, 0], [0]],
            pad_token_id=0,
        )

        self.assertEqual(batch["input_ids"].tolist(), [[101, 102], [101, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["token_type_ids"].tolist(), [[0, 0], [0, 0]])

    def test_pool_last_hidden_state_supports_cls_and_mean_pooling(self):
        hidden = np.array(
            [
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [9.0, 9.0],
                ]
            ],
            dtype=np.float32,
        )
        mask = np.array([[1, 1, 0]], dtype=np.int64)

        cls = pool_last_hidden_state(hidden, mask, pooling="cls")
        mean = pool_last_hidden_state(hidden, mask, pooling="mean")

        self.assertTrue(np.allclose(cls, np.array([[1.0, 0.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(mean, np.array([[0.5, 1.0]], dtype=np.float32)))

    def test_pool_torch_last_hidden_state_supports_cls_and_mean_pooling(self):
        hidden = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [9.0, 9.0],
                ]
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)

        cls = pool_torch_last_hidden_state(hidden, mask, torch_module=torch, pooling="cls")
        mean = pool_torch_last_hidden_state(hidden, mask, torch_module=torch, pooling="mean")

        self.assertTrue(torch.allclose(cls, torch.tensor([[1.0, 0.0]])))
        self.assertTrue(torch.allclose(mean, torch.tensor([[0.5, 1.0]])))


if __name__ == "__main__":
    unittest.main()
