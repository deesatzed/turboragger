from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from turboragger.contracts import RetrievalResult


RankedList = Sequence[RetrievalResult | tuple[str, float] | dict]


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, RankedList],
    k: int = 60,
    limit: int | None = None,
) -> list[RetrievalResult]:
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, list[str]] = defaultdict(list)

    for source, raw_results in ranked_lists.items():
        seen: set[str] = set()
        for rank, raw_result in enumerate(raw_results, start=1):
            doc_id = _doc_id(raw_result)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
            sources[doc_id].append(source)

    fused = [
        RetrievalResult(doc_id=doc_id, score=score, source="+".join(sources[doc_id]))
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda result: (-result.score, result.doc_id))
    if limit is not None:
        return fused[:limit]
    return fused


def _doc_id(result: RetrievalResult | tuple[str, float] | dict) -> str:
    if isinstance(result, RetrievalResult):
        return result.doc_id
    if isinstance(result, tuple):
        return str(result[0])
    if isinstance(result, dict) and "doc_id" in result:
        return str(result["doc_id"])
    raise ValueError(f"Unsupported ranked result: {result!r}")
