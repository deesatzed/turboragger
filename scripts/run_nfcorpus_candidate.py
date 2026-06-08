#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from turboragger.benchmark import build_candidate_payload, write_candidate_run, write_leaderboard
from turboragger.calibration import calibrate_rank_score_fusion_parameters, calibrate_score_fusion_weights
from turboragger.data import find_nfcorpus, load_nfcorpus, load_nfcorpus_qrels
from turboragger.dense import (
    MiniLMDenseRetriever,
    OnnxDenseRetriever,
    TransformerDenseRetriever,
    find_cached_minilm_snapshot,
)
from turboragger.harness import RetrievalHarness
from turboragger.late_interaction import OnnxLateInteractionReranker
from turboragger.leann_bridge import LeannMiniLMRetriever
from turboragger.learned_fusion import (
    CascadeFusionRetriever,
    LinearFeatureFusionRetriever,
    ModelFeatureFusionRetriever,
    cascade_ranked_results,
    fit_gbdt_feature_fusion,
    fit_gbdt_regression_feature_fusion,
    fit_linear_feature_fusion,
)
from turboragger.lexical import BM25PrfRetriever, BM25Retriever
from turboragger.metrics import score_ranked_results
from turboragger.probe import _probe_module
from turboragger.score_fusion import ScoreFusionRetriever


MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_LARGE_ZH_MODEL = "BAAI/bge-large-zh-v1.5"
BGE_LARGE_ZH_PATH = Path("/Volumes/WS4TB/WS4TBr/aP2A/ragflow/huggingface.co/BAAI/bge-large-zh-v1.5")
BCE_EMBEDDING_MODEL = "maidalun1020/bce-embedding-base_v1"
BCE_EMBEDDING_PATH = Path(
    "/Volumes/WS4TB/WS4TBr/aP2A/ragflow/huggingface.co/maidalun1020/bce-embedding-base_v1"
)
BGE_SMALL_EN_ONNX_MODEL = "Xenova/bge-small-en-v1.5"
BGE_SMALL_EN_ONNX_PATH = Path(
    "/Volumes/WS4TB/WS4TBr/finESS/node_modules/@xenova/transformers/.cache/Xenova/bge-small-en-v1.5"
)
XENOVA_MINILM_ONNX_MODEL = "Xenova/all-MiniLM-L6-v2"
XENOVA_MINILM_ONNX_PATH = Path(
    "/Volumes/WS4TB/WS4TBr/whsjan14/node_modules/@xenova/transformers/.cache/Xenova/all-MiniLM-L6-v2"
)
TOP_K = 100
CALIBRATION_GRID_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)
RANK_WEIGHT_GRID_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run named NFCorpus retrieval candidates.")
    parser.add_argument(
        "--candidate",
        choices=[
            "bm25",
            "bm25_prf",
            "minilm_dense",
            "minilm_bm25_rrf",
            "bge_large_zh",
            "bce_embedding_base_v1",
            "bge_small_en_onnx",
            "bge_small_en_bm25_rrf",
            "bge_small_minilm_bm25_rrf",
            "bge_small_minilm_bm25_prf_rrf",
            "bge_small_minilm_bm25_score_fusion",
            "xenova_minilm_onnx",
            "bge_small_xenova_minilm_bm25_score_fusion",
            "bge_small_late_interaction_score_fusion_rerank",
            "bge_small_mean_xenova_minilm_bm25_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade",
            "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion",
            "bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion",
            "bge_small_xenova_minilm_bm25_deep_score_fusion",
            "bge_small_xenova_minilm_bm25_mnz_fusion",
            "bge_small_dual_minilm_bm25_score_fusion",
            "leann_minilm_no_recompute",
            "all-baselines",
        ],
        required=True,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    candidates = ["bm25", "minilm_dense", "minilm_bm25_rrf"] if args.candidate == "all-baselines" else [args.candidate]
    command = f"PYTHONPATH=src python3 scripts/run_nfcorpus_candidate.py --candidate {args.candidate}"

    dataset = find_nfcorpus(root)
    if dataset["status"] != "found":
        print(json.dumps({"status": "blocked", "reason": "nfcorpus dataset not found", "dataset": dataset}, indent=2))
        return 2

    dataset_path = Path(dataset["path"])
    corpus, queries, qrels = load_nfcorpus(dataset_path)
    query_ids = [query_id for query_id in queries if query_id in qrels]
    filtered_queries = {query_id: queries[query_id] for query_id in query_ids}
    filtered_qrels = {query_id: qrels[query_id] for query_id in query_ids}

    written = []
    for candidate in candidates:
        payload = run_candidate(
            candidate_name=candidate,
            root=root,
            corpus=corpus,
            queries=filtered_queries,
            qrels=filtered_qrels,
            dataset=dataset,
            dataset_path=dataset_path,
            all_queries=queries,
            command=command,
        )
        path = write_candidate_run(root, payload)
        written.append(str(path.relative_to(root)))

    leaderboard_path = write_leaderboard(root)
    print(json.dumps({"written": written, "leaderboard": str(leaderboard_path.relative_to(root))}, indent=2))
    return 0


def run_candidate(
    *,
    candidate_name: str,
    root: Path,
    corpus: dict[str, dict],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    dataset: dict[str, Any],
    dataset_path: Path,
    all_queries: dict[str, str],
    command: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    run_id = f"{_slug(candidate_name)}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    calibration_qrels = load_nfcorpus_qrels(dataset_path, split="dev")
    calibration_queries = {
        query_id: all_queries[query_id]
        for query_id in all_queries
        if query_id in calibration_qrels
    }
    training_qrels = load_nfcorpus_qrels(dataset_path, split="train")
    training_queries = {
        query_id: all_queries[query_id]
        for query_id in all_queries
        if query_id in training_qrels
    }
    retrievers, retrieval_mode, retriever_config, dependency, hyperparameters = build_candidate(
        candidate_name,
        corpus,
        calibration_queries=calibration_queries,
        calibration_qrels=calibration_qrels,
        training_queries=training_queries,
        training_qrels=training_qrels,
    )

    harness = RetrievalHarness(retrievers)
    report = harness.run_with_report(queries, k=TOP_K)
    scores = score_ranked_results(report["runs"], qrels)
    runtime_seconds = round(time.perf_counter() - start, 6)

    return build_candidate_payload(
        run_id=run_id,
        candidate_name=candidate_name,
        retrieval_mode=retrieval_mode,
        dataset=dataset,
        retriever_config=retriever_config,
        dependency=dependency,
        report=report,
        scores=scores,
        command=command,
        hyperparameters=hyperparameters,
        runtime_seconds=runtime_seconds,
    )


def calibrate_candidate_weights(
    *,
    retrievers: dict[str, Any],
    calibration_queries: dict[str, str],
    calibration_qrels: dict[str, dict[str, int]],
    branch_names: list[str],
    grid_values: tuple[float, ...] = CALIBRATION_GRID_VALUES,
    k: int = TOP_K,
) -> dict[str, Any]:
    if not calibration_queries or not calibration_qrels:
        raise ValueError("Calibration queries and qrels are required.")

    branch_outputs: dict[str, dict[str, list]] = {}
    for query_id, query in calibration_queries.items():
        branch_outputs[query_id] = {}
        for branch_name, retriever in retrievers.items():
            branch_outputs[query_id][branch_name] = retriever.retrieve(query, k)

    weight_grid = [
        dict(zip(branch_names, values, strict=True))
        for values in _weight_product(grid_values, len(branch_names))
        if any(value > 0.0 for value in values)
    ]
    calibration = calibrate_score_fusion_weights(
        branch_outputs,
        calibration_qrels,
        weight_grid=weight_grid,
        limit=k,
    )
    return {
        **calibration,
        "split": "dev",
        "grid_values": list(grid_values),
    }


def calibrate_candidate_rank_score(
    *,
    retrievers: dict[str, Any],
    calibration_queries: dict[str, str],
    calibration_qrels: dict[str, dict[str, int]],
    weights: dict[str, float],
    rank_weight_grid: tuple[float, ...] = RANK_WEIGHT_GRID_VALUES,
    rrf_k: int = 60,
    k: int = TOP_K,
) -> dict[str, Any]:
    if not calibration_queries or not calibration_qrels:
        raise ValueError("Calibration queries and qrels are required.")

    branch_outputs: dict[str, dict[str, list]] = {}
    for query_id, query in calibration_queries.items():
        branch_outputs[query_id] = {}
        for branch_name, retriever in retrievers.items():
            branch_outputs[query_id][branch_name] = retriever.retrieve(query, k)

    calibration = calibrate_rank_score_fusion_parameters(
        branch_outputs,
        calibration_qrels,
        weights=weights,
        rank_weight_grid=rank_weight_grid,
        rrf_k=rrf_k,
        limit=k,
    )
    return {
        **calibration,
        "split": "dev",
    }


def fit_candidate_linear_fusion(
    *,
    retrievers: dict[str, Any],
    training_queries: dict[str, str],
    training_qrels: dict[str, dict[str, int]],
    branch_names: list[str],
    rrf_k: int = 60,
    k: int = TOP_K,
) -> dict[str, Any]:
    if not training_queries or not training_qrels:
        raise ValueError("Training queries and qrels are required.")

    branch_outputs: dict[str, dict[str, list]] = {}
    for query_id, query in training_queries.items():
        branch_outputs[query_id] = {}
        for branch_name, retriever in retrievers.items():
            branch_outputs[query_id][branch_name] = retriever.retrieve(query, k)

    fit = fit_linear_feature_fusion(
        branch_outputs,
        training_qrels,
        branch_names=branch_names,
        rrf_k=rrf_k,
    )
    return {
        **fit,
        "split": "train",
        "dependency": {"sklearn": _probe_module("sklearn")},
    }


def fit_candidate_gbdt_fusion(
    *,
    retrievers: dict[str, Any],
    training_queries: dict[str, str],
    training_qrels: dict[str, dict[str, int]],
    branch_names: list[str],
    rrf_k: int = 60,
    k: int = TOP_K,
) -> dict[str, Any]:
    if not training_queries or not training_qrels:
        raise ValueError("Training queries and qrels are required.")

    branch_outputs: dict[str, dict[str, list]] = {}
    for query_id, query in training_queries.items():
        branch_outputs[query_id] = {}
        for branch_name, retriever in retrievers.items():
            branch_outputs[query_id][branch_name] = retriever.retrieve(query, k)

    fit = fit_gbdt_feature_fusion(
        branch_outputs,
        training_qrels,
        branch_names=branch_names,
        rrf_k=rrf_k,
    )
    return {
        **fit,
        "split": "train",
        "dependency": {"sklearn": _probe_module("sklearn")},
    }


def fit_candidate_gbdt_regression_fusion(
    *,
    retrievers: dict[str, Any],
    training_queries: dict[str, str],
    training_qrels: dict[str, dict[str, int]],
    branch_names: list[str],
    rrf_k: int = 60,
    k: int = TOP_K,
) -> dict[str, Any]:
    if not training_queries or not training_qrels:
        raise ValueError("Training queries and qrels are required.")

    branch_outputs: dict[str, dict[str, list]] = {}
    for query_id, query in training_queries.items():
        branch_outputs[query_id] = {}
        for branch_name, retriever in retrievers.items():
            branch_outputs[query_id][branch_name] = retriever.retrieve(query, k)

    fit = fit_gbdt_regression_feature_fusion(
        branch_outputs,
        training_qrels,
        branch_names=branch_names,
        rrf_k=rrf_k,
    )
    return {
        **fit,
        "split": "train",
        "dependency": {"sklearn": _probe_module("sklearn")},
    }


def calibrate_candidate_cascade_anchor(
    *,
    primary_retriever: Any,
    secondary_retriever: Any,
    calibration_queries: dict[str, str],
    calibration_qrels: dict[str, dict[str, int]],
    anchor_grid: tuple[int, ...] = (0, 3, 5, 10, 20),
    k: int = TOP_K,
) -> dict[str, Any]:
    if not calibration_queries or not calibration_qrels:
        raise ValueError("Calibration queries and qrels are required.")
    primary_outputs: dict[str, list] = {}
    secondary_outputs: dict[str, list] = {}
    for query_id, query in calibration_queries.items():
        primary_outputs[query_id] = primary_retriever.retrieve(query, k)
        secondary_outputs[query_id] = secondary_retriever.retrieve(query, k)

    scored = []
    for anchor_k in anchor_grid:
        runs = {
            query_id: [
                result.doc_id
                for result in cascade_ranked_results(
                    primary_outputs[query_id],
                    secondary_outputs[query_id],
                    anchor_k=int(anchor_k),
                    limit=k,
                )
            ]
            for query_id in calibration_queries
        }
        scores = score_ranked_results(runs, calibration_qrels)
        scored.append(
            {
                "anchor_k": int(anchor_k),
                "metrics": scores["metrics"],
                "scores": scores,
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item["metrics"].get("nDCG@10", 0.0)),
            -float(item["metrics"].get("Recall@100", 0.0)),
            int(item["anchor_k"]),
        )
    )
    best = scored[0]
    return {
        "anchor_k": best["anchor_k"],
        "anchor_grid": [int(value) for value in anchor_grid],
        "metrics": best["metrics"],
        "query_count": best["scores"]["queries_tested"],
        "failure_count": best["scores"]["failure_count"],
        "grid_size": len(anchor_grid),
        "split": "dev",
        "all_results": [
            {"anchor_k": item["anchor_k"], "metrics": item["metrics"]}
            for item in scored
        ],
    }


def build_candidate(
    candidate_name: str,
    corpus: dict[str, dict],
    calibration_queries: dict[str, str] | None = None,
    calibration_qrels: dict[str, dict[str, int]] | None = None,
    training_queries: dict[str, str] | None = None,
    training_qrels: dict[str, dict[str, int]] | None = None,
) -> tuple[dict, str, dict, dict, dict]:
    rank_bm25 = _probe_module("rank_bm25")
    sentence_transformers = _probe_module("sentence_transformers")
    minilm_snapshot = find_cached_minilm_snapshot()

    if candidate_name == "bm25":
        return (
            {"bm25": BM25Retriever(corpus, source="bm25")},
            "lexical",
            {"model": "rank_bm25.BM25Okapi", "mode": "bm25", "top_k": TOP_K},
            {"rank_bm25": rank_bm25},
            {"k": TOP_K},
        )

    if candidate_name == "bm25_prf":
        return (
            {
                "bm25_prf": BM25PrfRetriever(
                    corpus,
                    source="bm25_prf",
                    feedback_docs=10,
                    expansion_terms=20,
                    expansion_repetitions=1,
                )
            },
            "lexical_prf",
            {
                "model": "rank_bm25.BM25Okapi",
                "mode": "bm25_pseudo_relevance_feedback",
                "top_k": TOP_K,
                "feedback_docs": 10,
                "expansion_terms": 20,
                "expansion_repetitions": 1,
            },
            {"rank_bm25": rank_bm25},
            {"k": TOP_K, "feedback_docs": 10, "expansion_terms": 20, "expansion_repetitions": 1},
        )

    if candidate_name == "minilm_dense":
        if minilm_snapshot is None:
            raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
        return (
            {"minilm_dense": MiniLMDenseRetriever(corpus, model_path=minilm_snapshot)},
            "dense",
            {
                "model": MINILM_MODEL,
                "model_path": str(minilm_snapshot),
                "mode": "dense_only_direct_transformers_mean_pooling",
                "top_k": TOP_K,
                "max_length": 256,
            },
            {"sentence_transformers": sentence_transformers, "direct_transformers": {"usable": True}},
            {"k": TOP_K, "max_length": 256},
        )

    if candidate_name == "minilm_bm25_rrf":
        if minilm_snapshot is None:
            raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
        return (
            {
                "minilm_dense": MiniLMDenseRetriever(corpus, model_path=minilm_snapshot),
                "bm25": BM25Retriever(corpus, source="bm25"),
            },
            "dense_sparse_rrf",
            {
                "model": f"{MINILM_MODEL} + rank_bm25.BM25Okapi",
                "model_path": str(minilm_snapshot),
                "mode": "minilm_dense_plus_bm25_rrf",
                "top_k": TOP_K,
                "rrf_k": 60,
                "max_length": 256,
            },
            {
                "sentence_transformers": sentence_transformers,
                "direct_transformers": {"usable": True},
                "rank_bm25": rank_bm25,
            },
            {"k": TOP_K, "rrf_k": 60, "max_length": 256},
        )

    if candidate_name == "bge_large_zh":
        if not BGE_LARGE_ZH_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE path not found: {BGE_LARGE_ZH_PATH}")
        return (
            {
                "bge_large_zh": TransformerDenseRetriever(
                    corpus,
                    model_path=BGE_LARGE_ZH_PATH,
                    source="bge_large_zh_direct_transformers",
                    batch_size=32,
                    max_length=256,
                )
            },
            "dense",
            {
                "model": BGE_LARGE_ZH_MODEL,
                "model_path": str(BGE_LARGE_ZH_PATH),
                "mode": "dense_only_direct_transformers_mean_pooling",
                "top_k": TOP_K,
                "max_length": 256,
                "note": "Local BGE-family fallback discovered on disk; Chinese-focused model, benchmarked because preferred BGE-M3/Qwen/Nomic/E5/GTE snapshots were unavailable.",
            },
            {"direct_transformers": {"usable": True}},
            {"k": TOP_K, "max_length": 256, "batch_size": 32},
        )

    if candidate_name == "bce_embedding_base_v1":
        if not BCE_EMBEDDING_PATH.is_dir():
            raise FileNotFoundError(f"Local BCE embedding path not found: {BCE_EMBEDDING_PATH}")
        return (
            {
                "bce_embedding_base_v1": TransformerDenseRetriever(
                    corpus,
                    model_path=BCE_EMBEDDING_PATH,
                    source="bce_embedding_base_v1_direct_transformers",
                    batch_size=32,
                    max_length=512,
                    pooling="cls",
                )
            },
            "dense",
            {
                "model": BCE_EMBEDDING_MODEL,
                "model_path": str(BCE_EMBEDDING_PATH),
                "mode": "dense_only_direct_transformers_cls_pooling",
                "top_k": TOP_K,
                "max_length": 512,
                "pooling": "cls",
                "note": "Complete local BCE embedding snapshot discovered under sibling ragflow cache; benchmarked as an unmeasured stronger multilingual embedding candidate.",
            },
            {"direct_transformers": {"usable": True}},
            {"k": TOP_K, "max_length": 512, "batch_size": 32, "pooling": "cls"},
        )

    if candidate_name == "bge_small_en_onnx":
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        return (
            {
                "bge_small_en_onnx": OnnxDenseRetriever(
                    corpus,
                    model_path=BGE_SMALL_EN_ONNX_PATH,
                    source="bge_small_en_onnx",
                    batch_size=64,
                    max_length=512,
                    pooling="cls",
                    query_prefix="Represent this sentence for searching relevant passages: ",
                )
            },
            "dense_onnx",
            {
                "model": BGE_SMALL_EN_ONNX_MODEL,
                "model_path": str(BGE_SMALL_EN_ONNX_PATH),
                "mode": "dense_only_onnxruntime_cls_pooling",
                "top_k": TOP_K,
                "max_length": 512,
                "pooling": "cls",
                "query_prefix": "Represent this sentence for searching relevant passages: ",
                "note": "Local English BGE-small ONNX fallback discovered in sibling finESS cache.",
            },
            {"onnxruntime": _probe_module("onnxruntime"), "tokenizers": _probe_module("tokenizers")},
            {"k": TOP_K, "max_length": 512, "batch_size": 64, "pooling": "cls"},
        )

    if candidate_name == "bge_small_en_bm25_rrf":
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        return (
            {
                "bge_small_en_onnx": OnnxDenseRetriever(
                    corpus,
                    model_path=BGE_SMALL_EN_ONNX_PATH,
                    source="bge_small_en_onnx",
                    batch_size=64,
                    max_length=512,
                    pooling="cls",
                    query_prefix="Represent this sentence for searching relevant passages: ",
                ),
                "bm25": BM25Retriever(corpus, source="bm25"),
            },
            "dense_onnx_sparse_rrf",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_path": str(BGE_SMALL_EN_ONNX_PATH),
                "mode": "bge_small_en_onnx_plus_bm25_rrf",
                "top_k": TOP_K,
                "rrf_k": 60,
                "max_length": 512,
                "pooling": "cls",
                "query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {"k": TOP_K, "rrf_k": 60, "max_length": 512, "batch_size": 64, "pooling": "cls"},
        )

    if candidate_name == "bge_small_minilm_bm25_rrf":
        if minilm_snapshot is None:
            raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        return (
            {
                "bge_small_en_onnx": OnnxDenseRetriever(
                    corpus,
                    model_path=BGE_SMALL_EN_ONNX_PATH,
                    source="bge_small_en_onnx",
                    batch_size=64,
                    max_length=512,
                    pooling="cls",
                    query_prefix="Represent this sentence for searching relevant passages: ",
                ),
                "minilm_dense": MiniLMDenseRetriever(corpus, model_path=minilm_snapshot),
                "bm25": BM25Retriever(corpus, source="bm25"),
            },
            "multi_dense_sparse_rrf",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + {MINILM_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "minilm_dense": str(minilm_snapshot),
                },
                "mode": "bge_small_en_onnx_plus_minilm_plus_bm25_rrf",
                "top_k": TOP_K,
                "rrf_k": 60,
                "bge_max_length": 512,
                "minilm_max_length": 256,
                "bge_pooling": "cls",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "sentence_transformers": sentence_transformers,
                "direct_transformers": {"usable": True},
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "rrf_k": 60,
                "bge_max_length": 512,
                "minilm_max_length": 256,
                "bge_batch_size": 64,
                "bge_pooling": "cls",
            },
        )

    if candidate_name == "bge_small_minilm_bm25_prf_rrf":
        if minilm_snapshot is None:
            raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        return (
            {
                "bge_small_en_onnx": OnnxDenseRetriever(
                    corpus,
                    model_path=BGE_SMALL_EN_ONNX_PATH,
                    source="bge_small_en_onnx",
                    batch_size=64,
                    max_length=512,
                    pooling="cls",
                    query_prefix="Represent this sentence for searching relevant passages: ",
                ),
                "minilm_dense": MiniLMDenseRetriever(corpus, model_path=minilm_snapshot),
                "bm25_prf": BM25PrfRetriever(
                    corpus,
                    source="bm25_prf",
                    feedback_docs=10,
                    expansion_terms=20,
                    expansion_repetitions=1,
                ),
            },
            "multi_dense_sparse_prf_rrf",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + {MINILM_MODEL} + BM25 PRF",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "minilm_dense": str(minilm_snapshot),
                },
                "mode": "bge_small_en_onnx_plus_minilm_plus_bm25_prf_rrf",
                "top_k": TOP_K,
                "rrf_k": 60,
                "bge_max_length": 512,
                "minilm_max_length": 256,
                "bm25_feedback_docs": 10,
                "bm25_expansion_terms": 20,
                "bge_pooling": "cls",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "sentence_transformers": sentence_transformers,
                "direct_transformers": {"usable": True},
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "rrf_k": 60,
                "bge_max_length": 512,
                "minilm_max_length": 256,
                "bge_batch_size": 64,
                "bge_pooling": "cls",
                "bm25_feedback_docs": 10,
                "bm25_expansion_terms": 20,
                "bm25_expansion_repetitions": 1,
            },
        )

    if candidate_name == "bge_small_minilm_bm25_score_fusion":
        if minilm_snapshot is None:
            raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_en_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_en_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "minilm_dense": MiniLMDenseRetriever(corpus, model_path=minilm_snapshot),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {"bge_small_en_onnx": 1.0, "minilm_dense": 1.0, "bm25": 1.0}
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source="bge_small_minilm_bm25_score_fusion",
                    weights=weights,
                )
            },
            "multi_dense_sparse_score_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + {MINILM_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "minilm_dense": str(minilm_snapshot),
                },
                "mode": "bge_small_en_onnx_plus_minilm_plus_bm25_score_fusion",
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "minilm_max_length": 256,
                "bge_pooling": "cls",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "sentence_transformers": sentence_transformers,
                "direct_transformers": {"usable": True},
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "minilm_max_length": 256,
                "bge_batch_size": 64,
                "bge_pooling": "cls",
            },
        )

    if candidate_name == "xenova_minilm_onnx":
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        return (
            {
                "xenova_minilm_onnx": OnnxDenseRetriever(
                    corpus,
                    model_path=XENOVA_MINILM_ONNX_PATH,
                    source="xenova_minilm_onnx",
                    batch_size=64,
                    max_length=256,
                    pooling="mean",
                )
            },
            "dense_onnx",
            {
                "model": XENOVA_MINILM_ONNX_MODEL,
                "model_path": str(XENOVA_MINILM_ONNX_PATH),
                "mode": "dense_only_onnxruntime_mean_pooling",
                "top_k": TOP_K,
                "max_length": 256,
                "pooling": "mean",
                "note": "Local ONNX MiniLM fallback discovered in a sibling Xenova transformers cache.",
            },
            {"onnxruntime": _probe_module("onnxruntime"), "tokenizers": _probe_module("tokenizers")},
            {"k": TOP_K, "max_length": 256, "batch_size": 64, "pooling": "mean"},
        )

    if candidate_name == "leann_minilm_no_recompute":
        if minilm_snapshot is None:
            raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
        index_path = Path("artifacts/leann_indexes/leann_minilm_no_recompute")
        return (
            {
                "leann_minilm_no_recompute": LeannMiniLMRetriever(
                    corpus,
                    model_path=minilm_snapshot,
                    index_path=index_path,
                )
            },
            "leann_dense_hnsw_no_recompute",
            {
                "model": MINILM_MODEL,
                "model_path": str(minilm_snapshot),
                "index_path": str(index_path),
                "mode": "leann_hnsw_minilm_no_recompute_no_compact",
                "top_k": TOP_K,
                "is_recompute": False,
                "is_compact": False,
                "embedding_mode": "sentence-transformers",
            },
            {
                "leann": _probe_module("leann"),
                "sentence_transformers": sentence_transformers,
                "direct_local_snapshot": {"usable": True},
            },
            {"k": TOP_K, "is_recompute": False, "is_compact": False, "hnsw_backend": "hnsw"},
        )

    if candidate_name == "bge_small_xenova_minilm_bm25_score_fusion":
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_en_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_en_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {"bge_small_en_onnx": 1.0, "xenova_minilm_onnx": 1.0, "bm25": 1.0}
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source="bge_small_xenova_minilm_bm25_score_fusion",
                    weights=weights,
                )
            },
            "multi_dense_sparse_score_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_en_onnx_plus_xenova_minilm_onnx_plus_bm25_score_fusion",
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_late_interaction_score_fusion_rerank":
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {
            "bge_small_cls_onnx": 1.0,
            "bge_small_mean_onnx": 1.0,
            "xenova_minilm_onnx": 1.0,
            "bm25": 1.0,
        }
        base_retriever = ScoreFusionRetriever(
            branch_retrievers,
            source="bge_small_dual_pool_xenova_minilm_bm25_score_fusion_base",
            weights=weights,
        )
        return (
            {
                "late_interaction_rerank": OnnxLateInteractionReranker(
                    corpus,
                    model_path=BGE_SMALL_EN_ONNX_PATH,
                    base_retriever=base_retriever,
                    source="bge_small_late_interaction_score_fusion_rerank",
                    candidate_k=TOP_K,
                    batch_size=16,
                    max_query_length=64,
                    max_doc_length=192,
                    query_prefix="Represent this sentence for searching relevant passages: ",
                )
            },
            "multi_dense_sparse_late_interaction_rerank",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} token MaxSim rerank over {BGE_SMALL_EN_ONNX_MODEL} + {XENOVA_MINILM_ONNX_MODEL} + BM25 score fusion",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_late_interaction_rerank_over_score_fusion",
                "top_k": TOP_K,
                "base_candidate_k": TOP_K,
                "base_score_normalization": "minmax_per_branch",
                "base_weights": weights,
                "late_interaction": "mean_query_token_maxsim",
                "max_query_length": 64,
                "max_doc_length": 192,
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "base_candidate_k": TOP_K,
                "base_weights": weights,
                "late_interaction": "mean_query_token_maxsim",
                "max_query_length": 64,
                "max_doc_length": 192,
                "late_interaction_batch_size": 16,
            },
        )

    if candidate_name == "bge_small_xenova_minilm_bm25_mnz_fusion":
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_en_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_en_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {"bge_small_en_onnx": 1.0, "xenova_minilm_onnx": 1.0, "bm25": 1.0}
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source="bge_small_xenova_minilm_bm25_mnz_fusion",
                    weights=weights,
                    mode="mnz",
                )
            },
            "multi_dense_sparse_mnz_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_en_onnx_plus_xenova_minilm_onnx_plus_bm25_mnz_fusion",
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "score_fusion_mode": "mnz",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "score_fusion_mode": "mnz",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_mean_xenova_minilm_bm25_score_fusion":
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {"bge_small_mean_onnx": 1.0, "xenova_minilm_onnx": 1.0, "bm25": 1.0}
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source="bge_small_mean_xenova_minilm_bm25_score_fusion",
                    weights=weights,
                )
            },
            "multi_dense_sparse_score_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} mean-pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_mean_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_score_fusion",
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_pooling": "mean",
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_pooling": "mean",
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name in {
        "bge_small_dual_pool_xenova_minilm_bm25_score_fusion",
        "bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion",
    }:
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        use_rank_score = candidate_name.endswith("_rank_score_fusion")
        fusion_mode = "rank_score" if use_rank_score else "sum"
        retrieval_mode = "multi_dense_sparse_rank_score_fusion" if use_rank_score else "multi_dense_sparse_score_fusion"
        source = (
            "bge_small_dual_pool_xenova_minilm_bm25_rank_score_fusion"
            if use_rank_score
            else "bge_small_dual_pool_xenova_minilm_bm25_score_fusion"
        )
        config_mode = (
            "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_rank_score_fusion"
            if use_rank_score
            else "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_score_fusion"
        )
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {
            "bge_small_cls_onnx": 1.0,
            "bge_small_mean_onnx": 1.0,
            "xenova_minilm_onnx": 1.0,
            "bm25": 1.0,
        }
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source=source,
                    weights=weights,
                    mode=fusion_mode,
                    rank_weight=1.0,
                    rrf_k=60,
                )
            },
            retrieval_mode,
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": config_mode,
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "score_fusion_mode": fusion_mode,
                "rank_weight": 1.0,
                "rrf_k": 60,
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "score_fusion_mode": fusion_mode,
                "rank_weight": 1.0,
                "rrf_k": 60,
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion":
        if calibration_queries is None or calibration_qrels is None:
            raise ValueError("Dev calibration queries and qrels are required for this candidate.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {
            "bge_small_cls_onnx": 1.0,
            "bge_small_mean_onnx": 1.0,
            "xenova_minilm_onnx": 1.0,
            "bm25": 1.0,
        }
        calibration = calibrate_candidate_rank_score(
            retrievers=branch_retrievers,
            calibration_queries=calibration_queries,
            calibration_qrels=calibration_qrels,
            weights=weights,
        )
        rank_weight = float(calibration["rank_weight"])
        rrf_k = int(calibration["rrf_k"])
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source="bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_rank_score_fusion",
                    weights=weights,
                    mode="rank_score",
                    rank_weight=rank_weight,
                    rrf_k=rrf_k,
                )
            },
            "multi_dense_sparse_dev_calibrated_rank_score_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_dev_calibrated_rank_score_fusion",
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "score_fusion_mode": "rank_score",
                "calibration": calibration,
                "rank_weight": rank_weight,
                "rrf_k": rrf_k,
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "score_fusion_mode": "rank_score",
                "calibration_split": "dev",
                "calibration_metric": "nDCG@10",
                "calibration_query_count": calibration["query_count"],
                "calibration_grid_size": calibration["grid_size"],
                "rank_weight": rank_weight,
                "rank_weight_grid": calibration["rank_weight_grid"],
                "rrf_k": rrf_k,
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion":
        if training_queries is None or training_qrels is None:
            raise ValueError("Training queries and qrels are required for this candidate.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        branch_names = list(branch_retrievers)
        fit = fit_candidate_linear_fusion(
            retrievers=branch_retrievers,
            training_queries=training_queries,
            training_qrels=training_qrels,
            branch_names=branch_names,
        )
        fit_rrf_k = int(fit.get("rrf_k", 60))
        return (
            {
                "learned_fusion": LinearFeatureFusionRetriever(
                    branch_retrievers,
                    score_weights=fit["score_weights"],
                    rank_weights=fit["rank_weights"],
                    intercept=fit["intercept"],
                    rrf_k=fit_rrf_k,
                    source="bge_small_dual_pool_xenova_minilm_bm25_train_dev_learned_fusion",
                )
            },
            "multi_dense_sparse_train_dev_learned_linear_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi + LogisticRegression",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_train_dev_learned_linear_fusion",
                "top_k": TOP_K,
                "feature_fusion": "logistic_regression_over_score_rank_presence_features",
                "fit": fit,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
                **fit["dependency"],
            },
            {
                "k": TOP_K,
                "feature_fusion": "logistic_regression_over_score_rank_presence_features",
                "training_split": "train",
                "training_query_count": fit["train_query_count"],
                "training_row_count": fit["train_row_count"],
                "positive_row_count": fit["positive_row_count"],
                "score_weights": fit["score_weights"],
                "rank_weights": fit["rank_weights"],
                "intercept": fit["intercept"],
                "feature_names": fit["feature_names"],
                "rrf_k": fit_rrf_k,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name in {
        "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion",
        "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_fusion",
    }:
        if training_queries is None or training_qrels is None:
            raise ValueError("Training queries and qrels are required for this candidate.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        branch_names = list(branch_retrievers)
        use_regression = candidate_name.endswith("_gbdt_regression_fusion")
        fit_function = fit_candidate_gbdt_regression_fusion if use_regression else fit_candidate_gbdt_fusion
        fit = fit_function(
            retrievers=branch_retrievers,
            training_queries=training_queries,
            training_qrels=training_qrels,
            branch_names=branch_names,
        )
        fit_rrf_k = int(fit.get("rrf_k", 60))
        retrieval_mode = (
            "multi_dense_sparse_train_dev_gbdt_regression_feature_fusion"
            if use_regression
            else "multi_dense_sparse_train_dev_gbdt_feature_fusion"
        )
        config_mode = (
            "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_train_dev_gbdt_regression_feature_fusion"
            if use_regression
            else "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_train_dev_gbdt_feature_fusion"
        )
        feature_fusion_name = (
            "hist_gradient_boosting_regression_over_graded_score_rank_presence_features"
            if use_regression
            else "hist_gradient_boosting_over_score_rank_presence_features"
        )
        return (
            {
                "gbdt_fusion": ModelFeatureFusionRetriever(
                    branch_retrievers,
                    branch_names=branch_names,
                    model=fit["model"],
                    rrf_k=fit_rrf_k,
                    source=candidate_name,
                )
            },
            retrieval_mode,
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi + HistGradientBoostingClassifier",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": config_mode,
                "top_k": TOP_K,
                "feature_fusion": feature_fusion_name,
                "fit": {key: value for key, value in fit.items() if key != "model"},
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
                **fit["dependency"],
            },
            {
                "k": TOP_K,
                "feature_fusion": feature_fusion_name,
                "training_split": "train",
                "training_query_count": fit["train_query_count"],
                "training_row_count": fit["train_row_count"],
                "positive_row_count": fit["positive_row_count"],
                "algorithm": fit["algorithm"],
                "model_params": fit["model_params"],
                "feature_names": fit["feature_names"],
                "max_relevance_target": fit.get("max_relevance_target"),
                "positive_sample_weight": fit.get("positive_sample_weight"),
                "rrf_k": fit_rrf_k,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade":
        if training_queries is None or training_qrels is None:
            raise ValueError("Training queries and qrels are required for this candidate.")
        if calibration_queries is None or calibration_qrels is None:
            raise ValueError("Dev calibration queries and qrels are required for this candidate.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        branch_names = list(branch_retrievers)
        weights = {
            "bge_small_cls_onnx": 1.0,
            "bge_small_mean_onnx": 1.0,
            "xenova_minilm_onnx": 1.0,
            "bm25": 1.0,
        }
        fit = fit_candidate_gbdt_fusion(
            retrievers=branch_retrievers,
            training_queries=training_queries,
            training_qrels=training_qrels,
            branch_names=branch_names,
        )
        fit_rrf_k = int(fit.get("rrf_k", 60))
        primary_retriever = ScoreFusionRetriever(
            branch_retrievers,
            source="bge_small_dual_pool_xenova_minilm_bm25_score_fusion",
            weights=weights,
        )
        secondary_retriever = ModelFeatureFusionRetriever(
            branch_retrievers,
            branch_names=branch_names,
            model=fit["model"],
            rrf_k=fit_rrf_k,
            source="bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion",
        )
        calibration = calibrate_candidate_cascade_anchor(
            primary_retriever=primary_retriever,
            secondary_retriever=secondary_retriever,
            calibration_queries=calibration_queries,
            calibration_qrels=calibration_qrels,
        )
        anchor_k = int(calibration["anchor_k"])
        return (
            {
                "cascade": CascadeFusionRetriever(
                    primary=primary_retriever,
                    secondary=secondary_retriever,
                    anchor_k=anchor_k,
                    source="bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_cascade",
                )
            },
            "multi_dense_sparse_train_dev_gbdt_dev_cascade",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi + score-fusion/GBDT cascade",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_train_dev_gbdt_dev_cascade",
                "top_k": TOP_K,
                "cascade_primary": "bge_small_dual_pool_xenova_minilm_bm25_score_fusion",
                "cascade_secondary": "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_fusion",
                "calibration": calibration,
                "fit": {key: value for key, value in fit.items() if key != "model"},
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
                **fit["dependency"],
            },
            {
                "k": TOP_K,
                "cascade_primary": "score_fusion",
                "cascade_secondary": "gbdt_feature_fusion",
                "calibration_split": "dev",
                "calibration_metric": "nDCG@10",
                "calibration_query_count": calibration["query_count"],
                "calibration_grid_size": calibration["grid_size"],
                "anchor_k": anchor_k,
                "anchor_grid": calibration["anchor_grid"],
                "training_split": "train",
                "training_query_count": fit["train_query_count"],
                "training_row_count": fit["train_row_count"],
                "positive_row_count": fit["positive_row_count"],
                "algorithm": fit["algorithm"],
                "model_params": fit["model_params"],
                "feature_names": fit["feature_names"],
                "rrf_k": fit_rrf_k,
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name in {
        "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_score_fusion",
        "bge_small_dual_pool_xenova_minilm_bm25_deep_train_dev_gbdt_dev_score_fusion",
        "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_regression_dev_score_fusion",
    }:
        if training_queries is None or training_qrels is None:
            raise ValueError("Training queries and qrels are required for this candidate.")
        if calibration_queries is None or calibration_qrels is None:
            raise ValueError("Dev calibration queries and qrels are required for this candidate.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        branch_names = list(branch_retrievers)
        base_weights = {
            "bge_small_cls_onnx": 1.0,
            "bge_small_mean_onnx": 1.0,
            "xenova_minilm_onnx": 1.0,
            "bm25": 1.0,
        }
        use_regression = candidate_name.endswith("_gbdt_regression_dev_score_fusion")
        fit_function = fit_candidate_gbdt_regression_fusion if use_regression else fit_candidate_gbdt_fusion
        fit = fit_function(
            retrievers=branch_retrievers,
            training_queries=training_queries,
            training_qrels=training_qrels,
            branch_names=branch_names,
        )
        fit_rrf_k = int(fit.get("rrf_k", 60))
        use_deep_pool = candidate_name.startswith("bge_small_dual_pool_xenova_minilm_bm25_deep_")
        branch_k = 300 if use_deep_pool else None
        retrieval_mode = (
            "multi_dense_sparse_deep_train_dev_gbdt_dev_score_fusion"
            if use_deep_pool
            else "multi_dense_sparse_train_dev_gbdt_regression_dev_score_fusion"
            if use_regression
            else "multi_dense_sparse_train_dev_gbdt_dev_score_fusion"
        )
        config_mode = (
            "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_deep_train_dev_gbdt_dev_score_fusion"
            if use_deep_pool
            else "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_train_dev_gbdt_regression_dev_score_fusion"
            if use_regression
            else "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_train_dev_gbdt_dev_score_fusion"
        )
        primary_retriever = ScoreFusionRetriever(
            branch_retrievers,
            source="score_fusion_primary",
            weights=base_weights,
            branch_k=branch_k,
        )
        secondary_retriever = ModelFeatureFusionRetriever(
            branch_retrievers,
            branch_names=branch_names,
            model=fit["model"],
            rrf_k=fit_rrf_k,
            branch_k=branch_k,
            source="gbdt_regression_secondary" if use_regression else "gbdt_secondary",
        )
        fusion_retrievers = {
            "score_fusion_primary": primary_retriever,
            ("gbdt_regression_secondary" if use_regression else "gbdt_secondary"): secondary_retriever,
        }
        calibration = calibrate_candidate_weights(
            retrievers=fusion_retrievers,
            calibration_queries=calibration_queries,
            calibration_qrels=calibration_qrels,
            branch_names=list(fusion_retrievers),
        )
        weights = calibration["weights"]
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    fusion_retrievers,
                    source=candidate_name,
                    weights=weights,
                    branch_k=branch_k,
                )
            },
            retrieval_mode,
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi + score-fusion/GBDT dev-calibrated score fusion",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": config_mode,
                "top_k": TOP_K,
                "branch_k": branch_k,
                "score_normalization": "minmax_per_ranker",
                "fusion_rankers": list(fusion_retrievers),
                "calibration": calibration,
                "fit": {key: value for key, value in fit.items() if key != "model"},
                "base_branch_weights": base_weights,
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
                **fit["dependency"],
            },
            {
                "k": TOP_K,
                "branch_k": branch_k,
                "score_normalization": "minmax_per_ranker",
                "fusion_rankers": list(fusion_retrievers),
                "calibration_split": "dev",
                "calibration_metric": "nDCG@10",
                "calibration_query_count": calibration["query_count"],
                "calibration_grid_size": calibration["grid_size"],
                "weights": weights,
                "base_branch_weights": base_weights,
                "training_split": "train",
                "training_query_count": fit["train_query_count"],
                "training_row_count": fit["train_row_count"],
                "positive_row_count": fit["positive_row_count"],
                "algorithm": fit["algorithm"],
                "model_params": fit["model_params"],
                "feature_names": fit["feature_names"],
                "max_relevance_target": fit.get("max_relevance_target"),
                "positive_sample_weight": fit.get("positive_sample_weight"),
                "rrf_k": fit_rrf_k,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion":
        if training_queries is None or training_qrels is None:
            raise ValueError("Training queries and qrels are required for this candidate.")
        if calibration_queries is None or calibration_qrels is None:
            raise ValueError("Dev calibration queries and qrels are required for this candidate.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        source_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        source_branch_names = list(source_retrievers)
        fit = fit_candidate_gbdt_fusion(
            retrievers=source_retrievers,
            training_queries=training_queries,
            training_qrels=training_qrels,
            branch_names=source_branch_names,
        )
        fit_rrf_k = int(fit.get("rrf_k", 60))
        gbdt_retriever = ModelFeatureFusionRetriever(
            source_retrievers,
            branch_names=source_branch_names,
            model=fit["model"],
            rrf_k=fit_rrf_k,
            source="gbdt_feature_fusion",
        )
        fusion_retrievers = {
            **source_retrievers,
            "gbdt_feature_fusion": gbdt_retriever,
        }
        calibration = calibrate_candidate_weights(
            retrievers=fusion_retrievers,
            calibration_queries=calibration_queries,
            calibration_qrels=calibration_qrels,
            branch_names=list(fusion_retrievers),
        )
        weights = calibration["weights"]
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    fusion_retrievers,
                    source="bge_small_dual_pool_xenova_minilm_bm25_train_dev_gbdt_dev_calibrated_score_fusion",
                    weights=weights,
                )
            },
            "multi_dense_sparse_train_dev_gbdt_dev_calibrated_score_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi + GBDT branch dev-calibrated score fusion",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_plus_train_gbdt_dev_calibrated_score_fusion",
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "calibration": calibration,
                "fit": {key: value for key, value in fit.items() if key != "model"},
                "source_branches": source_branch_names,
                "fusion_branches": list(fusion_retrievers),
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
                **fit["dependency"],
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "calibration_split": "dev",
                "calibration_metric": "nDCG@10",
                "calibration_query_count": calibration["query_count"],
                "calibration_grid_size": calibration["grid_size"],
                "weights": weights,
                "source_branches": source_branch_names,
                "fusion_branches": list(fusion_retrievers),
                "training_split": "train",
                "training_query_count": fit["train_query_count"],
                "training_row_count": fit["train_row_count"],
                "positive_row_count": fit["positive_row_count"],
                "algorithm": fit["algorithm"],
                "model_params": fit["model_params"],
                "feature_names": fit["feature_names"],
                "rrf_k": fit_rrf_k,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name in {
        "bge_small_dual_pool_xenova_minilm_bm25_dev_calibrated_score_fusion",
        "bge_small_dual_pool_xenova_minilm_bm25_fields_dev_calibrated_score_fusion",
    }:
        if calibration_queries is None or calibration_qrels is None:
            raise ValueError("Dev calibration queries and qrels are required for this candidate.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        use_field_branches = candidate_name.endswith("_fields_dev_calibrated_score_fusion")
        retrieval_mode = (
            "multi_dense_sparse_fields_dev_calibrated_score_fusion"
            if use_field_branches
            else "multi_dense_sparse_dev_calibrated_score_fusion"
        )
        config_mode = (
            "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_fields_dev_calibrated_score_fusion"
            if use_field_branches
            else "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_bm25_dev_calibrated_score_fusion"
        )
        lexical_branches = ["bm25"]
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        if use_field_branches:
            branch_retrievers["bm25_title"] = BM25Retriever(corpus, source="bm25_title", field="title")
            branch_retrievers["bm25_text"] = BM25Retriever(corpus, source="bm25_text", field="text")
            lexical_branches.extend(["bm25_title", "bm25_text"])
        branch_names = list(branch_retrievers)
        calibration = calibrate_candidate_weights(
            retrievers=branch_retrievers,
            calibration_queries=calibration_queries,
            calibration_qrels=calibration_qrels,
            branch_names=branch_names,
        )
        weights = calibration["weights"]
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source=candidate_name,
                    weights=weights,
                )
            },
            retrieval_mode,
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": config_mode,
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "calibration": calibration,
                "weights": weights,
                "lexical_branches": lexical_branches,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "calibration_split": "dev",
                "calibration_metric": "nDCG@10",
                "calibration_query_count": calibration["query_count"],
                "calibration_grid_size": calibration["grid_size"],
                "weights": weights,
                "lexical_branches": lexical_branches,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name in {
        "bge_small_dual_pool_xenova_minilm_bm25_title_score_fusion",
        "bge_small_dual_pool_xenova_minilm_bm25_text_score_fusion",
    }:
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        lexical_field = "title" if "_title_" in candidate_name else "text"
        lexical_source = f"bm25_{lexical_field}"
        lexical_mode = f"multi_dense_sparse_{lexical_field}_score_fusion"
        fusion_source = f"bge_small_dual_pool_xenova_minilm_bm25_{lexical_field}_score_fusion"
        config_mode = (
            "bge_small_cls_mean_onnx_plus_xenova_minilm_onnx_plus_"
            f"bm25_plus_{lexical_field}_bm25_score_fusion"
        )
        branch_retrievers = {
            "bge_small_cls_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_cls_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "bge_small_mean_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_mean_onnx",
                batch_size=64,
                max_length=512,
                pooling="mean",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
            lexical_source: BM25Retriever(corpus, source=lexical_source, field=lexical_field),
        }
        weights = {
            "bge_small_cls_onnx": 1.0,
            "bge_small_mean_onnx": 1.0,
            "xenova_minilm_onnx": 1.0,
            "bm25": 1.0,
            lexical_source: 1.0,
        }
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source=fusion_source,
                    weights=weights,
                )
            },
            lexical_mode,
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} CLS+mean pooling + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi full/{lexical_field}",
                "model_paths": {
                    "bge_small_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": config_mode,
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "lexical_branches": ["bm25", lexical_source],
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "lexical_branches": ["bm25", lexical_source],
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_poolings": ["cls", "mean"],
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_xenova_minilm_bm25_deep_score_fusion":
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_k = 300
        branch_retrievers = {
            "bge_small_en_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_en_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {"bge_small_en_onnx": 1.0, "xenova_minilm_onnx": 1.0, "bm25": 1.0}
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source="bge_small_xenova_minilm_bm25_deep_score_fusion",
                    weights=weights,
                    branch_k=branch_k,
                )
            },
            "multi_dense_sparse_deep_score_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_en_onnx_plus_xenova_minilm_onnx_plus_bm25_deep_score_fusion",
                "top_k": TOP_K,
                "branch_k": branch_k,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "branch_k": branch_k,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
            },
        )

    if candidate_name == "bge_small_dual_minilm_bm25_score_fusion":
        if minilm_snapshot is None:
            raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
        if not BGE_SMALL_EN_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local BGE ONNX path not found: {BGE_SMALL_EN_ONNX_PATH}")
        if not XENOVA_MINILM_ONNX_PATH.is_dir():
            raise FileNotFoundError(f"Local Xenova MiniLM ONNX path not found: {XENOVA_MINILM_ONNX_PATH}")
        branch_retrievers = {
            "bge_small_en_onnx": OnnxDenseRetriever(
                corpus,
                model_path=BGE_SMALL_EN_ONNX_PATH,
                source="bge_small_en_onnx",
                batch_size=64,
                max_length=512,
                pooling="cls",
                query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "minilm_dense": MiniLMDenseRetriever(corpus, model_path=minilm_snapshot),
            "xenova_minilm_onnx": OnnxDenseRetriever(
                corpus,
                model_path=XENOVA_MINILM_ONNX_PATH,
                source="xenova_minilm_onnx",
                batch_size=64,
                max_length=256,
                pooling="mean",
            ),
            "bm25": BM25Retriever(corpus, source="bm25"),
        }
        weights = {
            "bge_small_en_onnx": 1.0,
            "minilm_dense": 1.0,
            "xenova_minilm_onnx": 1.0,
            "bm25": 1.0,
        }
        return (
            {
                "score_fusion": ScoreFusionRetriever(
                    branch_retrievers,
                    source="bge_small_dual_minilm_bm25_score_fusion",
                    weights=weights,
                )
            },
            "multi_dense_sparse_score_fusion",
            {
                "model": f"{BGE_SMALL_EN_ONNX_MODEL} + {MINILM_MODEL} + {XENOVA_MINILM_ONNX_MODEL} + rank_bm25.BM25Okapi",
                "model_paths": {
                    "bge_small_en_onnx": str(BGE_SMALL_EN_ONNX_PATH),
                    "minilm_dense": str(minilm_snapshot),
                    "xenova_minilm_onnx": str(XENOVA_MINILM_ONNX_PATH),
                },
                "mode": "bge_small_en_onnx_plus_direct_minilm_plus_xenova_minilm_onnx_plus_bm25_score_fusion",
                "top_k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "direct_minilm_max_length": 256,
                "xenova_minilm_max_length": 256,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
                "bge_query_prefix": "Represent this sentence for searching relevant passages: ",
            },
            {
                "onnxruntime": _probe_module("onnxruntime"),
                "tokenizers": _probe_module("tokenizers"),
                "sentence_transformers": sentence_transformers,
                "direct_transformers": {"usable": True},
                "rank_bm25": rank_bm25,
            },
            {
                "k": TOP_K,
                "score_normalization": "minmax_per_branch",
                "weights": weights,
                "bge_max_length": 512,
                "direct_minilm_max_length": 256,
                "xenova_minilm_max_length": 256,
                "bge_batch_size": 64,
                "xenova_minilm_batch_size": 64,
                "bge_pooling": "cls",
                "xenova_minilm_pooling": "mean",
            },
        )

    raise ValueError(f"Unsupported candidate: {candidate_name}")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _weight_product(values: tuple[float, ...], length: int):
    if length == 0:
        yield ()
        return
    for value in values:
        for suffix in _weight_product(values, length - 1):
            yield (value, *suffix)


if __name__ == "__main__":
    raise SystemExit(main())
