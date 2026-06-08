from __future__ import annotations

from collections import defaultdict
from typing import Literal, Mapping, Protocol

from turboragger.contracts import RetrievalResult, validate_ranked_results


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult] | list[dict]:
        ...


class ScoreFusionRetriever:
    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        source: str = "score_fusion",
        weights: Mapping[str, float] | None = None,
        mode: Literal["sum", "mnz", "rank_score"] = "sum",
        branch_k: int | None = None,
        rank_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        if not retrievers:
            raise ValueError("At least one retriever branch is required.")
        if mode not in {"sum", "mnz", "rank_score"}:
            raise ValueError(f"Unsupported score fusion mode: {mode}")
        if branch_k is not None and branch_k < 1:
            raise ValueError("branch_k must be at least 1.")
        if rank_weight < 0.0:
            raise ValueError("rank_weight must be non-negative.")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")
        self.retrievers = dict(retrievers)
        self.source = source
        self.weights = dict(weights or {})
        self.mode = mode
        self.branch_k = branch_k
        self.rank_weight = rank_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        scores: dict[str, float] = defaultdict(float)
        contributors: dict[str, int] = defaultdict(int)
        branch_limit = max(k, self.branch_k or k)
        for branch_name, retriever in self.retrievers.items():
            results = validate_ranked_results(retriever.retrieve(query, branch_limit))
            normalized = minmax_normalize_results(results)
            weight = float(self.weights.get(branch_name, 1.0))
            ranks = {result.doc_id: rank for rank, result in enumerate(results, start=1)}
            for doc_id, score in normalized.items():
                scores[doc_id] += weight * score
                if self.mode == "rank_score":
                    scores[doc_id] += self.rank_weight * weight / (self.rrf_k + ranks[doc_id])
                contributors[doc_id] += 1

        fused = [
            RetrievalResult(
                doc_id=doc_id,
                score=score * contributors[doc_id] if self.mode == "mnz" else score,
                source=self.source,
            )
            for doc_id, score in scores.items()
        ]
        fused.sort(key=lambda result: (-result.score, result.doc_id))
        return fused[:k]


def minmax_normalize_results(results: list[RetrievalResult]) -> dict[str, float]:
    if not results:
        return {}
    scores = [result.score for result in results]
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return {result.doc_id: 1.0 for result in results}
    return {
        result.doc_id: (result.score - min_score) / (max_score - min_score)
        for result in results
    }
