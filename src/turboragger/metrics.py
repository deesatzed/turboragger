from __future__ import annotations

import math
from typing import Mapping, Sequence


Qrels = Mapping[str, int | float]


def recall_at_k(ranked_doc_ids: Sequence[str], qrels: Qrels, k: int = 100) -> float:
    relevant_doc_ids = {doc_id for doc_id, relevance in qrels.items() if relevance > 0}
    if not relevant_doc_ids:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(retrieved & relevant_doc_ids) / len(relevant_doc_ids)


def ndcg_at_k(ranked_doc_ids: Sequence[str], qrels: Qrels, k: int = 10) -> float:
    gains = [float(qrels.get(doc_id, 0.0)) for doc_id in ranked_doc_ids[:k]]
    if not gains:
        return 0.0

    dcg = _dcg(gains)
    ideal_gains = sorted((float(value) for value in qrels.values()), reverse=True)[:k]
    ideal = _dcg(ideal_gains)
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def score_ranked_results(
    runs: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Qrels],
    k_recall: int = 100,
    k_ndcg: int = 10,
) -> dict:
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    failure_count = 0

    for query_id, query_qrels in qrels.items():
        ranked = runs.get(query_id)
        if ranked is None:
            failure_count += 1
            ranked = []
        recall_scores.append(recall_at_k(ranked, query_qrels, k=k_recall))
        ndcg_scores.append(ndcg_at_k(ranked, query_qrels, k=k_ndcg))

    query_count = len(qrels)
    return {
        "queries_tested": query_count,
        "failure_count": failure_count,
        "metrics": {
            "Recall@100": _mean(recall_scores),
            "nDCG@10": _mean(ndcg_scores),
        },
        "per_query_scores": {
            "recall@100": recall_scores,
            "ndcg@10": ndcg_scores,
        },
    }


def _dcg(gains: Sequence[float]) -> float:
    total = 0.0
    for index, gain in enumerate(gains):
        total += gain / math.log2(index + 2)
    return total


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
