#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from turboragger.artifacts import write_json_artifact, write_markdown_artifact
from turboragger.data import find_nfcorpus, load_nfcorpus
from turboragger.dense import MiniLMDenseRetriever, find_cached_minilm_snapshot
from turboragger.harness import RetrievalHarness
from turboragger.lexical import BM25Retriever
from turboragger.metrics import score_ranked_results
from turboragger.probe import _probe_module


COMMAND = "PYTHONPATH=src python3 scripts/run_nfcorpus_baseline.py"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dataset = find_nfcorpus(root)
    sentence_transformers = _probe_module("sentence_transformers")
    minilm_snapshot = find_cached_minilm_snapshot()

    if dataset["status"] != "found":
        reasons = []
        reasons.append("nfcorpus dataset was not found in configured local paths")

        payload = {
            "status": "blocked",
            "baseline": "MiniLM nfcorpus",
            "retriever_config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "mode": "dense_only",
                "top_k": 100,
            },
            "dataset": dataset,
            "dependency": {"sentence_transformers": sentence_transformers},
            "queries_tested": 0,
            "failure_count": 0,
            "metrics": {"Recall@100": None, "nDCG@10": None},
            "reasons": reasons,
        }
        write_json_artifact(root / "artifacts" / "baseline_minilm_nfcorpus.json", payload, command=COMMAND)
        write_markdown_artifact(
            root / "artifacts" / "baseline_status.md",
            "Baseline Status",
            [
                "Status: blocked",
                "",
                "Reasons:",
                *[f"- {reason}" for reason in reasons],
                "",
                "This is not a methodology result. It only records why Step 0 could not run in this environment.",
            ],
        )
        print(json.dumps(payload, indent=2))
        return 2

    corpus, queries, qrels = load_nfcorpus(Path(dataset["path"]))
    query_ids = [query_id for query_id in queries if query_id in qrels]
    filtered_queries = {query_id: queries[query_id] for query_id in query_ids}

    dense_error = None
    if minilm_snapshot is not None:
        try:
            harness = RetrievalHarness(
                {"minilm_dense": MiniLMDenseRetriever(corpus, model_path=minilm_snapshot)}
            )
            report = harness.run_with_report(filtered_queries, k=100)
            scores = score_ranked_results(report["runs"], {query_id: qrels[query_id] for query_id in query_ids})
            payload = {
                "status": "reproduced_direct_minilm",
                "baseline": "MiniLM nfcorpus direct transformers",
                "historical_baseline_not_reproduced": True,
                "retriever_config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "model_path": str(minilm_snapshot),
                    "mode": "dense_only_direct_transformers_mean_pooling",
                    "top_k": 100,
                    "max_length": 256,
                },
                "dataset": dataset,
                "dependency": {
                    "sentence_transformers": sentence_transformers,
                    "direct_transformers": {"usable": True},
                },
                "queries_tested": scores["queries_tested"],
                "failure_count": scores["failure_count"] + len(report["failures"]),
                "metrics": scores["metrics"],
                "per_query_scores": scores["per_query_scores"],
                "reasons": [
                    "sentence_transformers wrapper is unavailable, but cached MiniLM was loaded directly with transformers.",
                    "Optional kernels were disabled in-process to avoid the installed transformers/kernels compatibility bug.",
                ],
            }
            write_json_artifact(root / "artifacts" / "baseline_minilm_nfcorpus.json", payload, command=COMMAND)
            write_markdown_artifact(
                root / "artifacts" / "baseline_status.md",
                "Baseline Status",
                [
                    "Status: reproduced_direct_minilm",
                    "",
                    f"Dataset: `{dataset['path']}`",
                    f"Model path: `{minilm_snapshot}`",
                    f"Queries tested: {scores['queries_tested']}",
                    f"Recall@100: {scores['metrics']['Recall@100']}",
                    f"nDCG@10: {scores['metrics']['nDCG@10']}",
                    "",
                    "This is a MiniLM dense baseline using direct transformers mean pooling through the neutral harness.",
                    "It is not an exact reproduction of the historical newragcity integrated artifact.",
                    "The sentence_transformers wrapper remains unavailable in this environment.",
                ],
            )
            print(json.dumps(payload, indent=2))
            return 0
        except Exception as exc:
            dense_error = {"error_type": type(exc).__name__, "error": str(exc)}

    if not sentence_transformers["usable"]:
        harness = RetrievalHarness({"bm25_replacement": BM25Retriever(corpus, source="bm25_replacement")})
        report = harness.run_with_report(filtered_queries, k=100)
        scores = score_ranked_results(report["runs"], {query_id: qrels[query_id] for query_id in query_ids})
        reasons = [
            f"MiniLM unavailable because sentence_transformers failed: {sentence_transformers.get('error_type')}: {sentence_transformers.get('error')}",
            "Established BM25 lexical replacement baseline with the same metric surface.",
        ]
        payload = {
            "status": "replacement_baseline",
            "baseline": "BM25 nfcorpus replacement",
            "historical_baseline_not_reproduced": True,
            "retriever_config": {
                "model": "rank_bm25.BM25Okapi",
                "mode": "lexical_replacement",
                "top_k": 100,
            },
            "dataset": dataset,
            "dependency": {
                "sentence_transformers": sentence_transformers,
                "direct_transformers": {
                    "usable": False,
                    "minilm_snapshot": str(minilm_snapshot) if minilm_snapshot else None,
                    "error": dense_error,
                },
            },
            "queries_tested": scores["queries_tested"],
            "failure_count": scores["failure_count"] + len(report["failures"]),
            "metrics": scores["metrics"],
            "per_query_scores": scores["per_query_scores"],
            "reasons": reasons,
        }
        write_json_artifact(root / "artifacts" / "baseline_minilm_nfcorpus.json", payload, command=COMMAND)
        write_json_artifact(root / "artifacts" / "baseline_bm25_nfcorpus.json", payload, command=COMMAND)
        write_markdown_artifact(
            root / "artifacts" / "baseline_status.md",
            "Baseline Status",
            [
                "Status: replacement_baseline",
                "",
                f"Dataset: `{dataset['path']}`",
                f"Queries tested: {scores['queries_tested']}",
                f"Recall@100: {scores['metrics']['Recall@100']}",
                f"nDCG@10: {scores['metrics']['nDCG@10']}",
                "",
                "Reasons:",
                *[f"- {reason}" for reason in reasons],
                "",
                "This is not the MiniLM dense baseline. It is a clean replacement baseline until sentence_transformers is repaired.",
            ],
        )
        print(json.dumps(payload, indent=2))
        return 0

    payload = {
        "status": "ready_not_implemented",
        "baseline": "MiniLM nfcorpus",
        "retriever_config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "mode": "dense_only",
            "top_k": 100,
        },
        "dataset": dataset,
        "queries_tested": 0,
        "failure_count": 0,
        "metrics": {"Recall@100": None, "nDCG@10": None},
        "reasons": ["dataset and dependency probes passed, but dense retrieval execution is not implemented in this first scaffold"],
    }
    write_json_artifact(root / "artifacts" / "baseline_minilm_nfcorpus.json", payload, command=COMMAND)
    print(json.dumps(payload, indent=2))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
