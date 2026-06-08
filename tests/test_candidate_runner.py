from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from turboragger.contracts import RetrievalResult
from turboragger.late_interaction import OnnxLateInteractionReranker
from turboragger.leann_bridge import LeannMiniLMRetriever
from turboragger.learned_fusion import CascadeFusionRetriever, LinearFeatureFusionRetriever, ModelFeatureFusionRetriever
from turboragger.score_fusion import ScoreFusionRetriever


class StaticRetriever:
    def retrieve(self, query: str, k: int = 100):
        return [RetrievalResult("d1", 1.0, "static")][:k]


class RankedStaticRetriever:
    def __init__(self, doc_ids: list[str]):
        self.doc_ids = doc_ids

    def retrieve(self, query: str, k: int = 100):
        return [
            RetrievalResult(doc_id, 1.0 / (index + 1), "static")
            for index, doc_id in enumerate(self.doc_ids[:k])
        ]


def load_candidate_runner():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_nfcorpus_candidate.py"
    spec = importlib.util.spec_from_file_location("run_nfcorpus_candidate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CandidateRunnerTests(unittest.TestCase):
    def test_score_fusion_candidate_builds_single_fusion_branch(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "find_cached_minilm_snapshot", return_value=model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "MiniLMDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_minilm_bm25_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertIsInstance(retrievers["score_fusion"], ScoreFusionRetriever)
        self.assertEqual(config["mode"], "bge_small_en_onnx_plus_minilm_plus_bm25_score_fusion")
        self.assertEqual(config["score_normalization"], "minmax_per_branch")
        self.assertEqual(hyperparameters["weights"]["bm25"], 1.0)
        self.assertIn("onnxruntime", dependency)

    def test_xenova_minilm_score_fusion_candidate_uses_onnx_minilm_branch(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_xenova_minilm_bm25_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(config["mode"], "bge_small_en_onnx_plus_xenova_minilm_onnx_plus_bm25_score_fusion")
        self.assertEqual(config["model_paths"]["xenova_minilm_onnx"], str(model_path))
        self.assertEqual(hyperparameters["weights"]["xenova_minilm_onnx"], 1.0)
        self.assertIn("tokenizers", dependency)

    def test_bge_mean_pooling_score_fusion_candidate_sets_bge_pooling(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()) as onnx_dense,
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_mean_xenova_minilm_bm25_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        bge_call = onnx_dense.call_args_list[0]
        self.assertEqual(mode, "multi_dense_sparse_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(config["mode"], "bge_small_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_score_fusion")
        self.assertEqual(config["bge_pooling"], "mean")
        self.assertEqual(hyperparameters["bge_pooling"], "mean")
        self.assertEqual(bge_call.kwargs["pooling"], "mean")
        self.assertIn("tokenizers", dependency)

    def test_bce_embedding_candidate_uses_cls_pooling(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BCE_EMBEDDING_PATH", model_path),
                mock.patch.object(runner, "TransformerDenseRetriever", return_value=StaticRetriever()) as dense,
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bce_embedding_base_v1",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "dense")
        self.assertEqual(list(retrievers), ["bce_embedding_base_v1"])
        self.assertEqual(config["model"], "maidalun1020/bce-embedding-base_v1")
        self.assertEqual(config["pooling"], "cls")
        self.assertEqual(hyperparameters["pooling"], "cls")
        self.assertEqual(dense.call_args.kwargs["pooling"], "cls")
        self.assertIn("direct_transformers", dependency)

    def test_scifact_finetuned_minilm_candidate_uses_local_domain_checkpoint(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "SCIFACT_FINETUNED_MINILM_PATH", model_path),
                mock.patch.object(runner, "TransformerDenseRetriever", return_value=StaticRetriever()) as dense,
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "scifact_finetuned_minilm",
                    {"d1": {"title": "title", "text": "text"}},
                )

        dense.assert_called_once_with(
            {"d1": {"title": "title", "text": "text"}},
            model_path=model_path,
            source="scifact_finetuned_minilm_direct_transformers",
            batch_size=32,
            max_length=256,
            pooling="mean",
        )
        self.assertEqual(mode, "dense")
        self.assertEqual(list(retrievers), ["scifact_finetuned_minilm"])
        self.assertEqual(config["model"], runner.SCIFACT_FINETUNED_MINILM_MODEL)
        self.assertEqual(config["model_path"], str(model_path))
        self.assertEqual(config["pooling"], "mean")
        self.assertEqual(hyperparameters["pooling"], "mean")
        self.assertEqual(hyperparameters["max_length"], 256)
        self.assertIn("direct_transformers", dependency)

    def test_scifact_dev_selected_minilm_candidate_selects_checkpoint_on_dev_qrels(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weak_path = root / "checkpoint-9"
            strong_path = root / "checkpoint-18"
            weak_path.mkdir()
            strong_path.mkdir()
            retriever_by_path = {
                str(weak_path): RankedStaticRetriever(["bad", "good"]),
                str(strong_path): RankedStaticRetriever(["good", "bad"]),
            }

            def fake_dense(corpus, *, model_path, source, batch_size, max_length, pooling):
                return retriever_by_path[str(model_path)]

            with (
                mock.patch.object(
                    runner,
                    "SCIFACT_FINETUNED_MINILM_CANDIDATE_PATHS",
                    [("checkpoint-9", weak_path), ("checkpoint-18", strong_path)],
                ),
                mock.patch.object(runner, "TransformerDenseRetriever", side_effect=fake_dense),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "scifact_dev_selected_minilm",
                    {"good": {"title": "good", "text": ""}, "bad": {"title": "bad", "text": ""}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"good": 1}},
                )

        self.assertEqual(mode, "dense_dev_selected")
        self.assertEqual(list(retrievers), ["scifact_dev_selected_minilm"])
        self.assertIs(retrievers["scifact_dev_selected_minilm"], retriever_by_path[str(strong_path)])
        self.assertEqual(config["selection"]["split"], "dev")
        self.assertEqual(config["selection"]["selected_label"], "checkpoint-18")
        self.assertEqual(config["selected_model_path"], str(strong_path))
        self.assertEqual(hyperparameters["selected_checkpoint"], "checkpoint-18")
        self.assertEqual(hyperparameters["candidate_count"], 2)
        self.assertIn("direct_transformers", dependency)

    def test_bge_dual_pooling_score_fusion_candidate_uses_cls_and_mean_branches(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()) as onnx_dense,
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        bge_poolings = [call.kwargs["pooling"] for call in onnx_dense.call_args_list[:2]]
        self.assertEqual(mode, "multi_dense_sparse_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(bge_poolings, ["cls", "mean"])
        self.assertEqual(config["mode"], "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_score_fusion")
        self.assertEqual(config["bge_poolings"], ["cls", "mean"])
        self.assertEqual(hyperparameters["weights"]["bge_small_cls_onnx"], 1.0)
        self.assertEqual(hyperparameters["weights"]["bge_small_mean_onnx"], 1.0)
        self.assertIn("tokenizers", dependency)

    def test_bge_dual_pooling_rank_score_candidate_sets_rank_score_mode(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_rank_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(retrievers["score_fusion"].mode, "rank_score")
        self.assertEqual(config["score_fusion_mode"], "rank_score")
        self.assertEqual(hyperparameters["rank_weight"], 1.0)
        self.assertEqual(hyperparameters["rrf_k"], 60)
        self.assertIn("tokenizers", dependency)

    def test_bge_dual_pooling_dev_calibrated_rank_score_candidate_uses_dev_rank_weight(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_rank_score",
                    return_value={
                        "rank_weight": 2.0,
                        "rrf_k": 60,
                        "weights": {
                            "bge_small_cls_onnx": 1.0,
                            "bge_small_mean_onnx": 1.0,
                            "xenova_minilm_onnx": 1.0,
                            "bm25": 1.0,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                        "rank_weight_grid": [0.0, 2.0],
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_dev_calibrated_rank_score_fusion")
        self.assertEqual(retrievers["score_fusion"].mode, "rank_score")
        self.assertEqual(retrievers["score_fusion"].rank_weight, 2.0)
        self.assertEqual(config["calibration"]["rank_weight"], 2.0)
        self.assertEqual(hyperparameters["rank_weight"], 2.0)
        self.assertIn("rank_weight_grid", hyperparameters)

    def test_train_dev_learned_linear_fusion_candidate_uses_fitted_parameters(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_linear_fusion",
                    return_value={
                        "score_weights": {
                            "bge_small_cls_onnx": 1.0,
                            "bge_small_mean_onnx": 0.5,
                            "xenova_minilm_onnx": 0.25,
                            "bm25": 2.0,
                        },
                        "rank_weights": {
                            "bge_small_cls_onnx": 0.0,
                            "bge_small_mean_onnx": 0.5,
                            "xenova_minilm_onnx": 0.0,
                            "bm25": 1.0,
                        },
                        "intercept": -0.25,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_train_dev_learned_linear_fusion")
        self.assertIsInstance(retrievers["learned_fusion"], LinearFeatureFusionRetriever)
        self.assertEqual(config["mode"], "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_train_dev_learned_linear_fusion")
        self.assertEqual(config["fit"]["intercept"], -0.25)
        self.assertEqual(hyperparameters["score_weights"]["bm25"], 2.0)
        self.assertEqual(hyperparameters["rank_weights"]["bm25"], 1.0)
        self.assertIn("sklearn", dependency)

    def test_train_dev_gbdt_fusion_candidate_uses_fitted_model(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingClassifier",
                        "model_params": {"max_iter": 25},
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_train_dev_gbdt_feature_fusion")
        self.assertIsInstance(retrievers["gbdt_fusion"], ModelFeatureFusionRetriever)
        self.assertIs(retrievers["gbdt_fusion"].model, fitted_model)
        self.assertEqual(config["fit"]["algorithm"], "HistGradientBoostingClassifier")
        self.assertEqual(hyperparameters["model_params"], {"max_iter": 25})
        self.assertIn("sklearn", dependency)

    def test_train_dev_gbdt_regression_fusion_candidate_uses_fitted_regressor(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_regression_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingRegressor",
                        "model_params": {"max_iter": 25},
                        "max_relevance_target": 2.0,
                        "positive_sample_weight": 4.0,
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 2}},
                )

        self.assertEqual(mode, "multi_dense_sparse_train_dev_gbdt_regression_feature_fusion")
        self.assertIsInstance(retrievers["gbdt_fusion"], ModelFeatureFusionRetriever)
        self.assertIs(retrievers["gbdt_fusion"].model, fitted_model)
        self.assertEqual(config["fit"]["algorithm"], "HistGradientBoostingRegressor")
        self.assertEqual(hyperparameters["max_relevance_target"], 2.0)
        self.assertEqual(hyperparameters["positive_sample_weight"], 4.0)
        self.assertIn("sklearn", dependency)

    def test_train_dev_gbdt_cascade_candidate_uses_dev_selected_anchor(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingClassifier",
                        "model_params": {"max_iter": 25},
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_cascade_anchor",
                    return_value={
                        "anchor_k": 5,
                        "anchor_grid": [0, 5, 10],
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "query_count": 1,
                        "grid_size": 3,
                        "split": "dev",
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 1}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_train_dev_gbdt_dev_cascade")
        self.assertIsInstance(retrievers["cascade"], CascadeFusionRetriever)
        self.assertEqual(retrievers["cascade"].anchor_k, 5)
        self.assertEqual(config["calibration"]["anchor_k"], 5)
        self.assertEqual(hyperparameters["anchor_k"], 5)
        self.assertIn("anchor_grid", hyperparameters)

    def test_train_dev_gbdt_score_fusion_candidate_uses_dev_weights(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingClassifier",
                        "model_params": {"max_iter": 25},
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    return_value={
                        "weights": {
                            "score_fusion_primary": 2.0,
                            "gbdt_secondary": 0.5,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 1}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_train_dev_gbdt_dev_score_fusion")
        self.assertIsInstance(retrievers["score_fusion"], ScoreFusionRetriever)
        self.assertEqual(retrievers["score_fusion"].weights["score_fusion_primary"], 2.0)
        self.assertEqual(retrievers["score_fusion"].weights["gbdt_secondary"], 0.5)
        self.assertEqual(config["calibration"]["weights"]["score_fusion_primary"], 2.0)
        self.assertEqual(hyperparameters["weights"]["gbdt_secondary"], 0.5)

    def test_deep_train_dev_gbdt_score_fusion_candidate_sets_branch_k(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingClassifier",
                        "model_params": {"max_iter": 25},
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    return_value={
                        "weights": {
                            "score_fusion_primary": 2.0,
                            "gbdt_secondary": 0.5,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 1}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_deep_train_dev_gbdt_dev_score_fusion")
        self.assertEqual(retrievers["score_fusion"].branch_k, 300)
        self.assertEqual(config["branch_k"], 300)
        self.assertEqual(hyperparameters["branch_k"], 300)
        self.assertIn("deep_train_dev_gbdt_dev_score_fusion", config["mode"])
        self.assertIn("sklearn", dependency)

    def test_train_dev_gbdt_regression_score_fusion_candidate_uses_regressor_fit(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_regression_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingRegressor",
                        "model_params": {"max_iter": 25},
                        "max_relevance_target": 2.0,
                        "positive_sample_weight": 4.0,
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    return_value={
                        "weights": {
                            "score_fusion_primary": 1.5,
                            "gbdt_regression_secondary": 2.0,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 2}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_train_dev_gbdt_regression_dev_score_fusion")
        self.assertIsInstance(retrievers["score_fusion"], ScoreFusionRetriever)
        self.assertEqual(retrievers["score_fusion"].weights["gbdt_regression_secondary"], 2.0)
        self.assertEqual(config["fit"]["algorithm"], "HistGradientBoostingRegressor")
        self.assertEqual(hyperparameters["max_relevance_target"], 2.0)
        self.assertEqual(hyperparameters["positive_sample_weight"], 4.0)
        self.assertIn("sklearn", dependency)

    def test_deep_train_dev_gbdt_regression_score_fusion_candidate_sets_branch_k(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_regression_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingRegressor",
                        "model_params": {"max_iter": 25},
                        "max_relevance_target": 2.0,
                        "positive_sample_weight": 4.0,
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    return_value={
                        "weights": {
                            "score_fusion_primary": 1.5,
                            "gbdt_regression_secondary": 2.0,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_regression_dev_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 2}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_deep_train_dev_gbdt_regression_dev_score_fusion")
        self.assertIsInstance(retrievers["score_fusion"], ScoreFusionRetriever)
        self.assertEqual(retrievers["score_fusion"].branch_k, 300)
        self.assertEqual(config["fit"]["algorithm"], "HistGradientBoostingRegressor")
        self.assertEqual(config["branch_k"], 300)
        self.assertEqual(hyperparameters["branch_k"], 300)
        self.assertIn("deep_train_dev_gbdt_regression_dev_score_fusion", config["mode"])
        self.assertIn("sklearn", dependency)

    def test_late_interaction_candidate_reranks_score_fusion_pool(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "OnnxLateInteractionReranker", return_value=StaticRetriever()) as reranker,
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_late_interaction_score_fusion_rerank",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_late_interaction_rerank")
        self.assertEqual(list(retrievers), ["late_interaction_rerank"])
        self.assertEqual(config["mode"], "bge_small_late_interaction_rerank_over_score_fusion")
        self.assertEqual(config["base_candidate_k"], 100)
        self.assertEqual(hyperparameters["max_doc_length"], 192)
        self.assertIn("onnxruntime", dependency)
        self.assertIsInstance(reranker.call_args.kwargs["base_retriever"], ScoreFusionRetriever)

    def test_late_interaction_gbdt_regression_dev_score_fusion_uses_dev_weights(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "OnnxLateInteractionReranker", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_regression_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingRegressor",
                        "model_params": {"max_iter": 25},
                        "max_relevance_target": 2.0,
                        "positive_sample_weight": 4.0,
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    side_effect=[
                        {
                            "weights": {
                                "score_fusion_primary": 1.0,
                                "gbdt_regression_secondary": 1.5,
                            },
                            "metrics": {"nDCG@10": 0.5, "Recall@100": 0.5},
                            "split": "dev",
                            "query_count": 1,
                            "grid_size": 2,
                        },
                        {
                            "weights": {
                                "current_best_primary": 2.0,
                                "late_interaction_secondary": 0.5,
                            },
                            "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                            "split": "dev",
                            "query_count": 1,
                            "grid_size": 2,
                        },
                    ],
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_late_interaction_gbdt_regression_dev_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 2}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_late_interaction_gbdt_regression_dev_score_fusion")
        self.assertIsInstance(retrievers["score_fusion"], ScoreFusionRetriever)
        self.assertEqual(retrievers["score_fusion"].weights["current_best_primary"], 2.0)
        self.assertEqual(retrievers["score_fusion"].weights["late_interaction_secondary"], 0.5)
        self.assertEqual(config["calibration"]["weights"]["late_interaction_secondary"], 0.5)
        self.assertEqual(hyperparameters["weights"]["current_best_primary"], 2.0)
        self.assertIn("sklearn", dependency)

    def test_train_dev_gbdt_calibrated_score_fusion_uses_gbdt_as_weighted_branch(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            fitted_model = object()
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "fit_candidate_gbdt_fusion",
                    return_value={
                        "model": fitted_model,
                        "feature_names": ["score:bge_small_cls_onnx"],
                        "train_query_count": 1,
                        "train_row_count": 2,
                        "positive_row_count": 1,
                        "algorithm": "HistGradientBoostingClassifier",
                        "model_params": {"max_iter": 25},
                        "dependency": {"sklearn": {"usable": True}},
                    },
                ),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    return_value={
                        "weights": {
                            "bge_small_cls_onnx": 0.5,
                            "bge_small_mean_onnx": 2.0,
                            "xenova_minilm_onnx": 1.5,
                            "bm25": 1.0,
                            "gbdt_feature_fusion": 2.0,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                    },
                ) as calibrate,
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    training_queries={"q1": "query"},
                    training_qrels={"q1": {"d1": 1}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        calibrated_branch_names = calibrate.call_args.kwargs["branch_names"]
        self.assertEqual(mode, "multi_dense_sparse_train_dev_gbdt_dev_calibrated_score_fusion")
        self.assertIsInstance(retrievers["score_fusion"], ScoreFusionRetriever)
        self.assertEqual(calibrated_branch_names[-1], "gbdt_feature_fusion")
        self.assertEqual(config["fusion_branches"], calibrated_branch_names)
        self.assertEqual(retrievers["score_fusion"].weights["gbdt_feature_fusion"], 2.0)
        self.assertEqual(hyperparameters["weights"]["bge_small_mean_onnx"], 2.0)
        self.assertIn("sklearn", dependency)

    def test_dev_calibrated_dual_pooling_candidate_uses_calibrated_weights(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    return_value={
                        "weights": {
                            "bge_small_cls_onnx": 1.0,
                            "bge_small_mean_onnx": 0.5,
                            "xenova_minilm_onnx": 1.5,
                            "bm25": 0.0,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        self.assertEqual(mode, "multi_dense_sparse_dev_calibrated_score_fusion")
        self.assertEqual(retrievers["score_fusion"].weights["bm25"], 0.0)
        self.assertEqual(config["calibration"]["split"], "dev")
        self.assertEqual(config["calibration"]["metrics"]["nDCG@10"], 1.0)
        self.assertEqual(hyperparameters["weights"]["xenova_minilm_onnx"], 1.5)

    def test_dev_calibrated_field_candidate_uses_title_and_text_bm25_branches(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()) as bm25,
                mock.patch.object(
                    runner,
                    "calibrate_candidate_weights",
                    return_value={
                        "weights": {
                            "bge_small_cls_onnx": 1.0,
                            "bge_small_mean_onnx": 0.5,
                            "xenova_minilm_onnx": 1.5,
                            "bm25": 1.0,
                            "bm25_title": 0.0,
                            "bm25_text": 0.5,
                        },
                        "metrics": {"nDCG@10": 1.0, "Recall@100": 1.0},
                        "split": "dev",
                        "query_count": 1,
                        "grid_size": 2,
                    },
                ),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                    calibration_queries={"q1": "query"},
                    calibration_qrels={"q1": {"d1": 1}},
                )

        bm25_fields = [call.kwargs.get("field", "all") for call in bm25.call_args_list]
        self.assertEqual(mode, "multi_dense_sparse_fields_dev_calibrated_score_fusion")
        self.assertEqual(config["lexical_branches"], ["bm25", "bm25_title", "bm25_text"])
        self.assertEqual(bm25_fields, ["all", "title", "text"])
        self.assertEqual(retrievers["score_fusion"].weights["bm25_title"], 0.0)
        self.assertEqual(hyperparameters["weights"]["bm25_text"], 0.5)
        self.assertIn("rank_bm25", dependency)

    def test_title_bm25_dual_pooling_candidate_adds_title_branch(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()) as bm25,
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_title_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(config["mode"], "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_plus_title_bm25_score_fusion")
        self.assertEqual(config["lexical_branches"], ["bm25", "bm25_title"])
        self.assertEqual(hyperparameters["weights"]["bm25_title"], 1.0)
        self.assertEqual(bm25.call_args_list[-1].kwargs["field"], "title")

    def test_text_bm25_dual_pooling_candidate_adds_text_branch(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()) as bm25,
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_text_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(config["mode"], "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_plus_text_bm25_score_fusion")
        self.assertEqual(config["lexical_branches"], ["bm25", "bm25_text"])
        self.assertEqual(hyperparameters["weights"]["bm25_text"], 1.0)
        self.assertEqual(bm25.call_args_list[-1].kwargs["field"], "text")

    def test_xenova_minilm_mnz_fusion_candidate_sets_mnz_mode(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_xenova_minilm_bm25_mnz_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_mnz_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(retrievers["score_fusion"].mode, "mnz")
        self.assertEqual(config["mode"], "bge_small_en_onnx_plus_xenova_minilm_onnx_plus_bm25_mnz_fusion")
        self.assertEqual(config["score_fusion_mode"], "mnz")
        self.assertEqual(hyperparameters["score_fusion_mode"], "mnz")
        self.assertIn("tokenizers", dependency)

    def test_xenova_minilm_deep_score_fusion_candidate_sets_branch_k(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_xenova_minilm_bm25_deep_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_deep_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(retrievers["score_fusion"].branch_k, 300)
        self.assertEqual(config["mode"], "bge_small_en_onnx_plus_xenova_minilm_onnx_plus_bm25_deep_score_fusion")
        self.assertEqual(config["branch_k"], 300)
        self.assertEqual(hyperparameters["branch_k"], 300)
        self.assertIn("tokenizers", dependency)

    def test_dual_minilm_score_fusion_candidate_uses_both_minilm_branches(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "BGE_SMALL_EN_ONNX_PATH", model_path),
                mock.patch.object(runner, "XENOVA_MINILM_ONNX_PATH", model_path),
                mock.patch.object(runner, "find_cached_minilm_snapshot", return_value=model_path),
                mock.patch.object(runner, "OnnxDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "MiniLMDenseRetriever", return_value=StaticRetriever()),
                mock.patch.object(runner, "BM25Retriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "bge_small_dual_minilm_bm25_score_fusion",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "multi_dense_sparse_score_fusion")
        self.assertEqual(list(retrievers), ["score_fusion"])
        self.assertEqual(config["mode"], "bge_small_en_onnx_plus_direct_minilm_plus_xenova_minilm_onnx_plus_bm25_score_fusion")
        self.assertEqual(config["model_paths"]["minilm_dense"], str(model_path))
        self.assertEqual(config["model_paths"]["xenova_minilm_onnx"], str(model_path))
        self.assertEqual(hyperparameters["weights"]["minilm_dense"], 1.0)
        self.assertEqual(hyperparameters["weights"]["xenova_minilm_onnx"], 1.0)
        self.assertIn("direct_transformers", dependency)

    def test_leann_minilm_candidate_uses_no_recompute_index(self):
        runner = load_candidate_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            with (
                mock.patch.object(runner, "find_cached_minilm_snapshot", return_value=model_path),
                mock.patch.object(runner, "LeannMiniLMRetriever", return_value=StaticRetriever()),
            ):
                retrievers, mode, config, dependency, hyperparameters = runner.build_candidate(
                    "leann_minilm_no_recompute",
                    {"d1": {"title": "title", "text": "text"}},
                )

        self.assertEqual(mode, "leann_dense_hnsw_no_recompute")
        self.assertEqual(list(retrievers), ["leann_minilm_no_recompute"])
        self.assertEqual(config["mode"], "leann_hnsw_minilm_no_recompute_no_compact")
        self.assertEqual(config["model_path"], str(model_path))
        self.assertFalse(hyperparameters["is_recompute"])
        self.assertIn("leann", dependency)


if __name__ == "__main__":
    unittest.main()
