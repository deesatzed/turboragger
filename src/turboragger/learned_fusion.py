from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Protocol, Sequence

from turboragger.contracts import RetrievalResult, validate_ranked_results
from turboragger.score_fusion import minmax_normalize_results


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult] | list[dict]:
        ...


BranchOutputsByQuery = Mapping[str, Mapping[str, Sequence[RetrievalResult]]]
QrelsByQuery = Mapping[str, Mapping[str, int]]


class LinearFeatureFusionRetriever:
    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        *,
        score_weights: Mapping[str, float],
        rank_weights: Mapping[str, float],
        intercept: float = 0.0,
        rrf_k: int = 60,
        branch_k: int | None = None,
        source: str = "linear_feature_fusion",
    ):
        if not retrievers:
            raise ValueError("At least one retriever branch is required.")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")
        if branch_k is not None and branch_k < 1:
            raise ValueError("branch_k must be at least 1.")
        self.retrievers = dict(retrievers)
        self.score_weights = {branch: float(weight) for branch, weight in score_weights.items()}
        self.rank_weights = {branch: float(weight) for branch, weight in rank_weights.items()}
        self.intercept = float(intercept)
        self.rrf_k = int(rrf_k)
        self.branch_k = branch_k
        self.source = source

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        branch_limit = max(k, self.branch_k or k)
        per_branch = {
            branch_name: validate_ranked_results(retriever.retrieve(query, branch_limit))
            for branch_name, retriever in self.retrievers.items()
        }
        return fuse_linear_feature_results(
            per_branch,
            score_weights=self.score_weights,
            rank_weights=self.rank_weights,
            intercept=self.intercept,
            rrf_k=self.rrf_k,
            limit=k,
            source=self.source,
        )


class ModelFeatureFusionRetriever:
    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        *,
        branch_names: Sequence[str],
        model,
        rrf_k: int = 60,
        branch_k: int | None = None,
        source: str = "model_feature_fusion",
    ):
        if not retrievers:
            raise ValueError("At least one retriever branch is required.")
        if not branch_names:
            raise ValueError("At least one branch name is required.")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")
        if branch_k is not None and branch_k < 1:
            raise ValueError("branch_k must be at least 1.")
        self.retrievers = dict(retrievers)
        self.branch_names = list(branch_names)
        self.model = model
        self.rrf_k = int(rrf_k)
        self.branch_k = branch_k
        self.source = source

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        branch_limit = max(k, self.branch_k or k)
        per_branch = {
            branch_name: validate_ranked_results(retriever.retrieve(query, branch_limit))
            for branch_name, retriever in self.retrievers.items()
        }
        doc_features = _feature_map_for_query(
            per_branch,
            branch_names=self.branch_names,
            rrf_k=self.rrf_k,
        )
        doc_ids = sorted(doc_features)
        rows = [doc_features[doc_id] for doc_id in doc_ids]
        scores = _predict_positive_scores(self.model, rows)
        fused = [
            RetrievalResult(doc_id=doc_id, score=float(score), source=self.source)
            for doc_id, score in zip(doc_ids, scores, strict=True)
        ]
        fused.sort(key=lambda result: (-result.score, result.doc_id))
        return fused[:k]


class CascadeFusionRetriever:
    def __init__(
        self,
        *,
        primary: Retriever,
        secondary: Retriever,
        anchor_k: int,
        source: str = "cascade_fusion",
    ):
        if anchor_k < 0:
            raise ValueError("anchor_k must be non-negative.")
        self.primary = primary
        self.secondary = secondary
        self.anchor_k = int(anchor_k)
        self.source = source

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        primary_results = validate_ranked_results(self.primary.retrieve(query, k))
        secondary_results = validate_ranked_results(self.secondary.retrieve(query, k))
        return cascade_ranked_results(
            primary_results,
            secondary_results,
            anchor_k=self.anchor_k,
            limit=k,
            source=self.source,
        )


def cascade_ranked_results(
    primary_results: Sequence[RetrievalResult],
    secondary_results: Sequence[RetrievalResult],
    *,
    anchor_k: int,
    limit: int = 100,
    source: str = "cascade_fusion",
) -> list[RetrievalResult]:
    if anchor_k < 0:
        raise ValueError("anchor_k must be non-negative.")
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    primary = validate_ranked_results(primary_results)
    secondary = validate_ranked_results(secondary_results)
    ordered: list[RetrievalResult] = []
    seen: set[str] = set()

    for result in primary[:anchor_k]:
        if result.doc_id in seen:
            continue
        ordered.append(RetrievalResult(result.doc_id, float(len(primary) - len(ordered)), source))
        seen.add(result.doc_id)
        if len(ordered) >= limit:
            return ordered

    for result in secondary:
        if result.doc_id in seen:
            continue
        ordered.append(RetrievalResult(result.doc_id, float(len(primary) - len(ordered)), source))
        seen.add(result.doc_id)
        if len(ordered) >= limit:
            return ordered

    for result in primary:
        if result.doc_id in seen:
            continue
        ordered.append(RetrievalResult(result.doc_id, float(len(primary) - len(ordered)), source))
        seen.add(result.doc_id)
        if len(ordered) >= limit:
            return ordered

    return ordered


def fuse_linear_feature_results(
    per_branch: Mapping[str, Sequence[RetrievalResult]],
    *,
    score_weights: Mapping[str, float],
    rank_weights: Mapping[str, float],
    intercept: float = 0.0,
    rrf_k: int = 60,
    limit: int = 100,
    source: str = "linear_feature_fusion",
) -> list[RetrievalResult]:
    scores: dict[str, float] = defaultdict(lambda: float(intercept))
    seen_doc_ids: set[str] = set()
    for branch_name, raw_results in per_branch.items():
        results = validate_ranked_results(raw_results)
        normalized = minmax_normalize_results(results)
        score_weight = float(score_weights.get(branch_name, 0.0))
        rank_weight = float(rank_weights.get(branch_name, 0.0))
        for rank, result in enumerate(results, start=1):
            seen_doc_ids.add(result.doc_id)
            scores[result.doc_id] += score_weight * normalized.get(result.doc_id, 0.0)
            scores[result.doc_id] += rank_weight / (rrf_k + rank)

    fused = [
        RetrievalResult(doc_id=doc_id, score=scores[doc_id], source=source)
        for doc_id in seen_doc_ids
    ]
    fused.sort(key=lambda result: (-result.score, result.doc_id))
    return fused[:limit]


def build_labeled_feature_rows(
    branch_outputs: BranchOutputsByQuery,
    qrels: QrelsByQuery,
    *,
    branch_names: Sequence[str],
    rrf_k: int = 60,
) -> tuple[list[list[float]], list[int], list[str]]:
    feature_names = (
        [f"score:{branch_name}" for branch_name in branch_names]
        + [f"rank:{branch_name}" for branch_name in branch_names]
        + ["presence_count"]
    )
    rows: list[list[float]] = []
    labels: list[int] = []

    for query_id, per_branch in branch_outputs.items():
        doc_features = _feature_map_for_query(per_branch, branch_names=branch_names, rrf_k=rrf_k)
        relevant_docs = {
            doc_id
            for doc_id, relevance in qrels.get(query_id, {}).items()
            if relevance > 0
        }
        for doc_id in sorted(doc_features):
            rows.append(doc_features[doc_id])
            labels.append(1 if doc_id in relevant_docs else 0)

    return rows, labels, feature_names


def build_graded_feature_rows(
    branch_outputs: BranchOutputsByQuery,
    qrels: QrelsByQuery,
    *,
    branch_names: Sequence[str],
    rrf_k: int = 60,
) -> tuple[list[list[float]], list[float], list[str]]:
    feature_names = (
        [f"score:{branch_name}" for branch_name in branch_names]
        + [f"rank:{branch_name}" for branch_name in branch_names]
        + ["presence_count"]
    )
    rows: list[list[float]] = []
    targets: list[float] = []

    for query_id, per_branch in branch_outputs.items():
        doc_features = _feature_map_for_query(per_branch, branch_names=branch_names, rrf_k=rrf_k)
        query_qrels = qrels.get(query_id, {})
        for doc_id in sorted(doc_features):
            rows.append(doc_features[doc_id])
            targets.append(float(max(0, query_qrels.get(doc_id, 0))))

    return rows, targets, feature_names


def fit_linear_feature_fusion(
    branch_outputs: BranchOutputsByQuery,
    qrels: QrelsByQuery,
    *,
    branch_names: Sequence[str],
    rrf_k: int = 60,
) -> dict:
    rows, labels, feature_names = build_labeled_feature_rows(
        branch_outputs,
        qrels,
        branch_names=branch_names,
        rrf_k=rrf_k,
    )
    if not rows:
        raise ValueError("No training rows were generated.")
    if len(set(labels)) < 2:
        raise ValueError("Training rows must include both positive and negative labels.")

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=0,
    )
    model.fit(rows, labels)
    coefficients = [float(value) for value in model.coef_[0]]
    intercept = float(model.intercept_[0])
    score_weights = {
        branch_name: coefficients[index]
        for index, branch_name in enumerate(branch_names)
    }
    rank_offset = len(branch_names)
    rank_weights = {
        branch_name: coefficients[rank_offset + index]
        for index, branch_name in enumerate(branch_names)
    }
    return {
        "score_weights": score_weights,
        "rank_weights": rank_weights,
        "intercept": intercept,
        "feature_names": feature_names,
        "train_query_count": len(branch_outputs),
        "train_row_count": len(rows),
        "positive_row_count": sum(labels),
        "rrf_k": rrf_k,
    }


def fit_gbdt_feature_fusion(
    branch_outputs: BranchOutputsByQuery,
    qrels: QrelsByQuery,
    *,
    branch_names: Sequence[str],
    rrf_k: int = 60,
    max_iter: int = 80,
    max_leaf_nodes: int = 15,
    learning_rate: float = 0.05,
) -> dict:
    rows, labels, feature_names = build_labeled_feature_rows(
        branch_outputs,
        qrels,
        branch_names=branch_names,
        rrf_k=rrf_k,
    )
    if not rows:
        raise ValueError("No training rows were generated.")
    if len(set(labels)) < 2:
        raise ValueError("Training rows must include both positive and negative labels.")

    from sklearn.ensemble import HistGradientBoostingClassifier

    model_params = {
        "max_iter": int(max_iter),
        "max_leaf_nodes": int(max_leaf_nodes),
        "learning_rate": float(learning_rate),
        "l2_regularization": 0.01,
        "random_state": 0,
        "class_weight": "balanced",
    }
    model = HistGradientBoostingClassifier(**model_params)
    model.fit(rows, labels)
    return {
        "model": model,
        "algorithm": "HistGradientBoostingClassifier",
        "model_params": model_params,
        "feature_names": feature_names,
        "train_query_count": len(branch_outputs),
        "train_row_count": len(rows),
        "positive_row_count": sum(labels),
        "rrf_k": rrf_k,
    }


def fit_gbdt_regression_feature_fusion(
    branch_outputs: BranchOutputsByQuery,
    qrels: QrelsByQuery,
    *,
    branch_names: Sequence[str],
    rrf_k: int = 60,
    max_iter: int = 80,
    max_leaf_nodes: int = 15,
    learning_rate: float = 0.05,
) -> dict:
    rows, targets, feature_names = build_graded_feature_rows(
        branch_outputs,
        qrels,
        branch_names=branch_names,
        rrf_k=rrf_k,
    )
    if not rows:
        raise ValueError("No training rows were generated.")
    positive_row_count = sum(1 for target in targets if target > 0.0)
    if positive_row_count == 0:
        raise ValueError("Training rows must include positive relevance targets.")

    from sklearn.ensemble import HistGradientBoostingRegressor

    negative_row_count = len(targets) - positive_row_count
    positive_weight = min(20.0, max(1.0, negative_row_count / positive_row_count))
    sample_weight = [positive_weight if target > 0.0 else 1.0 for target in targets]
    model_params = {
        "max_iter": int(max_iter),
        "max_leaf_nodes": int(max_leaf_nodes),
        "learning_rate": float(learning_rate),
        "l2_regularization": 0.01,
        "random_state": 0,
    }
    model = HistGradientBoostingRegressor(**model_params)
    model.fit(rows, targets, sample_weight=sample_weight)
    return {
        "model": model,
        "algorithm": "HistGradientBoostingRegressor",
        "model_params": model_params,
        "feature_names": feature_names,
        "train_query_count": len(branch_outputs),
        "train_row_count": len(rows),
        "positive_row_count": positive_row_count,
        "max_relevance_target": max(targets),
        "positive_sample_weight": positive_weight,
        "rrf_k": rrf_k,
    }


def _feature_map_for_query(
    per_branch: Mapping[str, Sequence[RetrievalResult]],
    *,
    branch_names: Sequence[str],
    rrf_k: int,
) -> dict[str, list[float]]:
    feature_count = len(branch_names) * 2 + 1
    features: dict[str, list[float]] = defaultdict(lambda: [0.0] * feature_count)
    for branch_index, branch_name in enumerate(branch_names):
        results = validate_ranked_results(per_branch.get(branch_name, []))
        normalized = minmax_normalize_results(results)
        for rank, result in enumerate(results, start=1):
            row = features[result.doc_id]
            row[branch_index] = normalized.get(result.doc_id, 0.0)
            row[len(branch_names) + branch_index] = 1.0 / (rrf_k + rank)
            row[-1] += 1.0
    return dict(features)


def _predict_positive_scores(model, rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        return []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(rows)
        return [float(row[1]) for row in probabilities]
    if hasattr(model, "decision_function"):
        return [float(value) for value in model.decision_function(rows)]
    return [float(value) for value in model.predict(rows)]
