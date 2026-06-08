from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    score: float
    source: str = "unknown"


def validate_ranked_results(results: Iterable[RetrievalResult | dict[str, Any]]) -> list[RetrievalResult]:
    validated: list[RetrievalResult] = []
    for index, result in enumerate(results):
        if isinstance(result, RetrievalResult):
            validated.append(result)
            continue
        if not isinstance(result, dict):
            raise ValueError(f"Result {index} must be RetrievalResult or dict.")
        if "doc_id" not in result:
            raise ValueError(f"Result {index} missing doc_id.")
        if "score" not in result:
            raise ValueError(f"Result {index} missing score.")
        validated.append(
            RetrievalResult(
                doc_id=str(result["doc_id"]),
                score=float(result["score"]),
                source=str(result.get("source", "unknown")),
            )
        )
    return validated
