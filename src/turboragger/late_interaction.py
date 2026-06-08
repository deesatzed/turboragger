from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from turboragger.contracts import RetrievalResult, validate_ranked_results
from turboragger.dense import _load_onnx_model, pad_token_batch


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult] | list[dict]:
        ...


class OnnxLateInteractionReranker:
    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        model_path: Path,
        base_retriever: Retriever,
        source: str = "onnx_late_interaction_rerank",
        candidate_k: int = 100,
        batch_size: int = 16,
        max_query_length: int = 64,
        max_doc_length: int = 192,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")
        if max_query_length < 1 or max_doc_length < 1:
            raise ValueError("max token lengths must be at least 1.")
        self.corpus = corpus
        self.model_path = model_path
        self.onnx_path = model_path / "onnx" / "model_quantized.onnx"
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        self.base_retriever = base_retriever
        self.source = source
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.tokenizer, self.session = _load_onnx_model(self.model_path, self.onnx_path)
        self._doc_token_cache: dict[str, np.ndarray] = {}

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        limit = min(k, self.candidate_k)
        candidates = validate_ranked_results(self.base_retriever.retrieve(query, self.candidate_k))
        if not candidates:
            return []
        query_tokens = self._encode_texts([f"{self.query_prefix}{query}"], max_length=self.max_query_length)[0]
        self._populate_doc_cache([candidate.doc_id for candidate in candidates])
        doc_tokens = {
            candidate.doc_id: self._doc_token_cache[candidate.doc_id]
            for candidate in candidates
            if candidate.doc_id in self._doc_token_cache
        }
        return rerank_candidates_with_maxsim(
            candidates,
            query_tokens=query_tokens,
            doc_tokens_by_id=doc_tokens,
            source=self.source,
            limit=limit,
        )

    def _populate_doc_cache(self, doc_ids: list[str]) -> None:
        missing = [doc_id for doc_id in doc_ids if doc_id not in self._doc_token_cache and doc_id in self.corpus]
        if not missing:
            return
        texts = [
            f"{self.document_prefix}{self.corpus[doc_id].get('title', '')}\n{self.corpus[doc_id].get('text', '')}"
            for doc_id in missing
        ]
        encoded = self._encode_texts(texts, max_length=self.max_doc_length)
        for doc_id, tokens in zip(missing, encoded, strict=True):
            self._doc_token_cache[doc_id] = tokens

    def _encode_texts(self, texts: list[str], *, max_length: int) -> list[np.ndarray]:
        encoded_texts = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            encoded = [self.tokenizer.encode(text) for text in batch_texts]
            token_batch = pad_token_batch(
                input_ids=[item.ids[:max_length] for item in encoded],
                attention_masks=[item.attention_mask[:max_length] for item in encoded],
                token_type_ids=[item.type_ids[:max_length] for item in encoded],
                pad_token_id=0,
            )
            outputs = self.session.run(None, token_batch)
            hidden = _normalize_token_vectors(outputs[0].astype(np.float32))
            masks = token_batch["attention_mask"].astype(bool)
            for item_hidden, item_mask in zip(hidden, masks, strict=True):
                encoded_texts.append(item_hidden[item_mask])
        return encoded_texts


def maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return 0.0
    similarities = query_tokens @ doc_tokens.T
    return float(np.max(similarities, axis=1).mean())


def rerank_candidates_with_maxsim(
    candidates: list[RetrievalResult],
    *,
    query_tokens: np.ndarray,
    doc_tokens_by_id: Mapping[str, np.ndarray],
    source: str,
    limit: int,
) -> list[RetrievalResult]:
    scored = []
    base_count = len(candidates)
    for rank, candidate in enumerate(candidates, start=1):
        doc_tokens = doc_tokens_by_id.get(candidate.doc_id)
        if doc_tokens is None:
            continue
        score = maxsim_score(query_tokens, doc_tokens)
        rank_prior = 1e-9 * (base_count - rank)
        scored.append(RetrievalResult(candidate.doc_id, score + rank_prior, source))
    scored.sort(key=lambda result: (-result.score, result.doc_id))
    return scored[:limit]


def _normalize_token_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=2, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms
