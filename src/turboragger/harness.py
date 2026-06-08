from __future__ import annotations

from typing import Mapping, Protocol

from turboragger.contracts import RetrievalResult, validate_ranked_results
from turboragger.rrf import reciprocal_rank_fusion


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult] | list[dict]:
        ...


class RetrievalHarness:
    def __init__(self, retrievers: Mapping[str, Retriever]):
        if not retrievers:
            raise ValueError("At least one retriever branch is required.")
        self.retrievers = dict(retrievers)

    def run_queries(self, queries: Mapping[str, str], k: int = 100) -> dict[str, list[str]]:
        return self.run_with_report(queries, k=k)["runs"]

    def run_with_report(self, queries: Mapping[str, str], k: int = 100) -> dict:
        runs: dict[str, list[str]] = {}
        branch_outputs: dict[str, dict[str, list[dict]]] = {}
        failures: dict[str, str] = {}

        for query_id, query in queries.items():
            per_branch: dict[str, list[RetrievalResult]] = {}
            branch_outputs[query_id] = {}
            for branch_name, retriever in self.retrievers.items():
                try:
                    results = validate_ranked_results(retriever.retrieve(query, k))
                except Exception as exc:
                    failures[f"{query_id}:{branch_name}"] = f"{type(exc).__name__}: {exc}"
                    results = []
                per_branch[branch_name] = results
                branch_outputs[query_id][branch_name] = [
                    {"doc_id": result.doc_id, "score": result.score, "source": result.source}
                    for result in results
                ]

            fused = reciprocal_rank_fusion(per_branch, limit=k)
            runs[query_id] = [result.doc_id for result in fused]

        return {"runs": runs, "branch_outputs": branch_outputs, "failures": failures}
