from __future__ import annotations

from typing import Mapping, Sequence

from turboragger.contracts import RetrievalResult, validate_ranked_results
from turboragger.metrics import score_ranked_results
from turboragger.score_fusion import minmax_normalize_results


BranchResults = Mapping[str, Sequence[RetrievalResult]]
BranchOutputsByQuery = Mapping[str, Mapping[str, Sequence[RetrievalResult]]]
WeightSet = Mapping[str, float]


def fuse_score_results(
    per_branch: BranchResults,
    *,
    weights: WeightSet,
    limit: int = 100,
    source: str = "calibrated_score_fusion",
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    for branch_name, results in per_branch.items():
        normalized = minmax_normalize_results(validate_ranked_results(results))
        weight = float(weights.get(branch_name, 0.0))
        for doc_id, score in normalized.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * score

    fused = [
        RetrievalResult(doc_id=doc_id, score=score, source=source)
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda result: (-result.score, result.doc_id))
    return fused[:limit]


def fuse_rank_score_results(
    per_branch: BranchResults,
    *,
    weights: WeightSet,
    rank_weight: float,
    rrf_k: int = 60,
    limit: int = 100,
    source: str = "calibrated_rank_score_fusion",
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    for branch_name, raw_results in per_branch.items():
        results = validate_ranked_results(raw_results)
        normalized = minmax_normalize_results(results)
        weight = float(weights.get(branch_name, 0.0))
        ranks = {result.doc_id: rank for rank, result in enumerate(results, start=1)}
        for doc_id, score in normalized.items():
            rank_bonus = rank_weight * weight / (rrf_k + ranks[doc_id])
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * score + rank_bonus

    fused = [
        RetrievalResult(doc_id=doc_id, score=score, source=source)
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda result: (-result.score, result.doc_id))
    return fused[:limit]


def calibrate_score_fusion_weights(
    branch_outputs: BranchOutputsByQuery,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    weight_grid: Sequence[WeightSet],
    limit: int = 100,
) -> dict:
    if not weight_grid:
        raise ValueError("At least one weight set is required.")

    scored = []
    for weights in weight_grid:
        runs = {
            query_id: [
                result.doc_id
                for result in fuse_score_results(per_branch, weights=weights, limit=limit)
            ]
            for query_id, per_branch in branch_outputs.items()
        }
        scores = score_ranked_results(runs, qrels)
        scored.append({"weights": dict(weights), "metrics": scores["metrics"], "scores": scores})

    scored.sort(
        key=lambda item: (
            -float(item["metrics"].get("nDCG@10", 0.0)),
            -float(item["metrics"].get("Recall@100", 0.0)),
            _weight_tiebreak(item["weights"]),
        )
    )
    best = scored[0]
    return {
        "weights": best["weights"],
        "metrics": best["metrics"],
        "query_count": best["scores"]["queries_tested"],
        "failure_count": best["scores"]["failure_count"],
        "grid_size": len(weight_grid),
        "all_results": [
            {"weights": item["weights"], "metrics": item["metrics"]}
            for item in scored
        ],
    }


def calibrate_rank_score_fusion_parameters(
    branch_outputs: BranchOutputsByQuery,
    qrels: Mapping[str, Mapping[str, int]],
    *,
    weights: WeightSet,
    rank_weight_grid: Sequence[float],
    rrf_k: int = 60,
    limit: int = 100,
) -> dict:
    if not rank_weight_grid:
        raise ValueError("At least one rank_weight value is required.")

    scored = []
    for rank_weight in rank_weight_grid:
        runs = {
            query_id: [
                result.doc_id
                for result in fuse_rank_score_results(
                    per_branch,
                    weights=weights,
                    rank_weight=float(rank_weight),
                    rrf_k=rrf_k,
                    limit=limit,
                )
            ]
            for query_id, per_branch in branch_outputs.items()
        }
        scores = score_ranked_results(runs, qrels)
        scored.append(
            {
                "rank_weight": float(rank_weight),
                "rrf_k": rrf_k,
                "weights": dict(weights),
                "metrics": scores["metrics"],
                "scores": scores,
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item["metrics"].get("nDCG@10", 0.0)),
            -float(item["metrics"].get("Recall@100", 0.0)),
            float(item["rank_weight"]),
        )
    )
    best = scored[0]
    return {
        "rank_weight": best["rank_weight"],
        "rrf_k": best["rrf_k"],
        "weights": best["weights"],
        "metrics": best["metrics"],
        "query_count": best["scores"]["queries_tested"],
        "failure_count": best["scores"]["failure_count"],
        "grid_size": len(rank_weight_grid),
        "rank_weight_grid": [float(value) for value in rank_weight_grid],
        "all_results": [
            {
                "rank_weight": item["rank_weight"],
                "rrf_k": item["rrf_k"],
                "weights": item["weights"],
                "metrics": item["metrics"],
            }
            for item in scored
        ],
    }


def _weight_tiebreak(weights: Mapping[str, float]) -> tuple[float, tuple[tuple[str, float], ...]]:
    return (sum(float(value) for value in weights.values()), tuple(sorted(weights.items())))
