from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DOC_ID_KEYS = ("runs", "branch_outputs", "retrieved_docs", "retrieved_doc_ids")


def audit_newragcity_historical_claim(result_path: Path, source_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    source = source_path.read_text()
    metrics = dict(result.get("metrics", {}))
    per_query = dict(result.get("per_query_scores", {}))
    recall_scores = [float(score) for score in per_query.get("recall@100", [])]

    issues: list[str] = []
    if not _has_retrieved_doc_ids(result):
        issues.append("missing_retrieved_doc_ids")
    if any(score > 1.0 for score in recall_scores):
        issues.append("recall_above_one")
    if _uses_top10_only_idcg(source):
        issues.append("top10_only_idcg")
    if _has_recall_double_count_risk(source):
        issues.append("recall_double_count_risk")

    verdict = "invalid" if issues else "needs_rescore"
    return {
        "schema_version": 1,
        "claim": "newragcity historical BEIR NFCorpus unified result",
        "result_path": str(result_path),
        "source_path": str(source_path),
        "verdict": verdict,
        "claimed_metrics": metrics,
        "queries_tested": result.get("queries_tested"),
        "has_retrieved_doc_ids": _has_retrieved_doc_ids(result),
        "recall_gt_1_count": sum(1 for score in recall_scores if score > 1.0),
        "recall_max": max(recall_scores) if recall_scores else None,
        "issues": issues,
        "rescore_possible_from_saved_artifact": _has_retrieved_doc_ids(result),
        "conclusion": _conclusion(verdict, issues),
    }


def _has_retrieved_doc_ids(result: Mapping[str, Any]) -> bool:
    return any(key in result for key in DOC_ID_KEYS)


def _uses_top10_only_idcg(source: str) -> bool:
    return "ideal_relevances = sorted(relevances" in source


def _has_recall_double_count_risk(source: str) -> bool:
    return "retrieved_results[:10]" in source and "retrieved_results[:100]" in source


def _conclusion(verdict: str, issues: list[str]) -> str:
    if verdict == "invalid":
        return (
            "The saved historical score is not comparable SOTA evidence. "
            "It must be rerun through the root benchmark harness or another corrected metric implementation."
        )
    return "The saved artifact lacks enough evidence for acceptance until rerun or rescored."
