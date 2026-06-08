from __future__ import annotations

import builtins
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np

from turboragger.contracts import RetrievalResult


MINILM_CACHE_REPO = "models--sentence-transformers--all-MiniLM-L6-v2"


class DenseVectorIndex:
    def __init__(self, doc_ids: list[str], vectors: np.ndarray, source: str):
        if len(doc_ids) != len(vectors):
            raise ValueError("doc_ids and vectors must have the same length.")
        self.doc_ids = doc_ids
        self.vectors = _normalize(np.asarray(vectors, dtype=np.float32))
        self.source = source

    def search(self, query_vector: np.ndarray, k: int = 100) -> list[RetrievalResult]:
        query = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ query
        ranked_indices = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievalResult(doc_id=self.doc_ids[index], score=float(scores[index]), source=self.source)
            for index in ranked_indices
        ]


class TransformerDenseRetriever:
    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        model_path: Path,
        source: str,
        batch_size: int = 64,
        max_length: int = 256,
        pooling: Literal["cls", "mean"] = "mean",
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self.model_path = model_path
        self.source = source
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.tokenizer, self.model, self.torch = _load_transformers_model(self.model_path)
        self.doc_ids = list(corpus.keys())
        documents = [
            f"{self.document_prefix}{doc.get('title', '')}\n{doc.get('text', '')}"
            for doc in corpus.values()
        ]
        self.index = DenseVectorIndex(
            doc_ids=self.doc_ids,
            vectors=self.encode(documents),
            source=source,
        )

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        query_vector = self.encode([f"{self.query_prefix}{query}"])[0]
        return self.index.search(query_vector, k=k)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        all_vectors = []
        with self.torch.no_grad():
            for start in range(0, len(text_list), self.batch_size):
                batch_texts = text_list[start : start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                output = self.model(**batch)
                pooled = pool_torch_last_hidden_state(
                    output.last_hidden_state,
                    batch["attention_mask"],
                    torch_module=self.torch,
                    pooling=self.pooling,
                )
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vectors.append(pooled.cpu().numpy().astype(np.float32))
        if not all_vectors:
            return np.empty((0, 384), dtype=np.float32)
        return np.vstack(all_vectors)


class MiniLMDenseRetriever(TransformerDenseRetriever):
    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        model_path: Path | None = None,
        batch_size: int = 64,
        max_length: int = 256,
    ):
        super().__init__(
            corpus=corpus,
            model_path=model_path or require_cached_minilm_snapshot(),
            source="minilm_dense_direct_transformers",
            batch_size=batch_size,
            max_length=max_length,
        )


class DensePrfRetriever:
    def __init__(
        self,
        base_retriever: TransformerDenseRetriever | "OnnxDenseRetriever",
        source: str,
        feedback_docs: int = 10,
        query_weight: float = 1.0,
        feedback_weight: float = 0.5,
    ):
        if feedback_docs < 1:
            raise ValueError("feedback_docs must be at least 1.")
        if query_weight < 0.0:
            raise ValueError("query_weight must be non-negative.")
        if feedback_weight < 0.0:
            raise ValueError("feedback_weight must be non-negative.")
        self.base_retriever = base_retriever
        self.source = source
        self.feedback_docs = feedback_docs
        self.query_weight = query_weight
        self.feedback_weight = feedback_weight
        self._doc_index_by_id = {
            doc_id: index
            for index, doc_id in enumerate(base_retriever.index.doc_ids)
        }

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        query_text = f"{getattr(self.base_retriever, 'query_prefix', '')}{query}"
        query_vector = self.base_retriever.encode([query_text])[0]
        initial = self.base_retriever.index.search(query_vector, k=max(k, self.feedback_docs))
        feedback_vectors = np.asarray(
            [
                self.base_retriever.index.vectors[self._doc_index_by_id[result.doc_id]]
                for result in initial[: self.feedback_docs]
                if result.doc_id in self._doc_index_by_id
            ],
            dtype=np.float32,
        )
        expanded_query = dense_prf_query_vector(
            query_vector,
            feedback_vectors,
            query_weight=self.query_weight,
            feedback_weight=self.feedback_weight,
        )
        results = self.base_retriever.index.search(expanded_query, k=k)
        return [
            RetrievalResult(result.doc_id, result.score, self.source)
            for result in results
        ]


class OnnxDenseRetriever:
    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        model_path: Path,
        source: str,
        batch_size: int = 64,
        max_length: int = 512,
        pooling: Literal["cls", "mean"] = "cls",
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self.model_path = model_path
        self.onnx_path = model_path / "onnx" / "model_quantized.onnx"
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        self.source = source
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.tokenizer, self.session = _load_onnx_model(self.model_path, self.onnx_path)
        self.doc_ids = list(corpus.keys())
        documents = [
            f"{self.document_prefix}{doc.get('title', '')}\n{doc.get('text', '')}"
            for doc in corpus.values()
        ]
        self.index = DenseVectorIndex(
            doc_ids=self.doc_ids,
            vectors=self.encode(documents),
            source=source,
        )

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        query_vector = self.encode([f"{self.query_prefix}{query}"])[0]
        return self.index.search(query_vector, k=k)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        all_vectors = []
        for start in range(0, len(text_list), self.batch_size):
            batch_texts = text_list[start : start + self.batch_size]
            encoded = [self.tokenizer.encode(text) for text in batch_texts]
            token_batch = pad_token_batch(
                input_ids=[item.ids[: self.max_length] for item in encoded],
                attention_masks=[item.attention_mask[: self.max_length] for item in encoded],
                token_type_ids=[item.type_ids[: self.max_length] for item in encoded],
                pad_token_id=0,
            )
            outputs = self.session.run(None, token_batch)
            pooled = pool_last_hidden_state(outputs[0], token_batch["attention_mask"], pooling=self.pooling)
            all_vectors.append(_normalize(pooled.astype(np.float32)))
        if not all_vectors:
            return np.empty((0, 384), dtype=np.float32)
        return np.vstack(all_vectors)


def require_cached_minilm_snapshot(cache_root: str | Path | None = None) -> Path:
    snapshot = find_cached_minilm_snapshot(cache_root=cache_root)
    if snapshot is None:
        raise FileNotFoundError("Cached all-MiniLM-L6-v2 snapshot not found.")
    return snapshot


def find_cached_minilm_snapshot(cache_root: str | Path | None = None, include_default: bool = True) -> Path | None:
    roots = []
    if cache_root is not None:
        roots.append(Path(cache_root).expanduser())
    if include_default:
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in roots:
        repo_root = root / MINILM_CACHE_REPO
        snapshots = repo_root / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir()):
            if _is_complete_minilm_snapshot(snapshot):
                return snapshot
    return None


def _is_complete_minilm_snapshot(path: Path) -> bool:
    required = ["config.json", "model.safetensors", "tokenizer.json", "vocab.txt"]
    return path.is_dir() and all((path / name).is_file() for name in required)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def dense_prf_query_vector(
    query_vector: np.ndarray,
    feedback_vectors: np.ndarray,
    *,
    query_weight: float = 1.0,
    feedback_weight: float = 0.5,
) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    feedback = np.asarray(feedback_vectors, dtype=np.float32)
    if feedback.size == 0:
        return _normalize(query.reshape(1, -1))[0]
    if feedback.ndim == 1:
        feedback = feedback.reshape(1, -1)
    centroid = feedback.mean(axis=0)
    expanded = query_weight * query + feedback_weight * centroid
    return _normalize(expanded.reshape(1, -1))[0]


def pad_token_batch(
    *,
    input_ids: list[list[int]],
    attention_masks: list[list[int]],
    token_type_ids: list[list[int]],
    pad_token_id: int,
) -> dict[str, np.ndarray]:
    max_length = max((len(ids) for ids in input_ids), default=0)
    padded_ids = []
    padded_masks = []
    padded_types = []
    for ids, mask, types in zip(input_ids, attention_masks, token_type_ids, strict=True):
        pad_length = max_length - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_length)
        padded_masks.append(mask + [0] * pad_length)
        padded_types.append(types + [0] * pad_length)
    return {
        "input_ids": np.asarray(padded_ids, dtype=np.int64),
        "attention_mask": np.asarray(padded_masks, dtype=np.int64),
        "token_type_ids": np.asarray(padded_types, dtype=np.int64),
    }


def pool_last_hidden_state(
    last_hidden_state: np.ndarray,
    attention_mask: np.ndarray,
    pooling: Literal["cls", "mean"],
) -> np.ndarray:
    if pooling == "cls":
        return last_hidden_state[:, 0, :].astype(np.float32)
    if pooling == "mean":
        mask = attention_mask.astype(np.float32)[:, :, None]
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return (summed / counts).astype(np.float32)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def pool_torch_last_hidden_state(last_hidden_state, attention_mask, *, torch_module, pooling: Literal["cls", "mean"]):
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def _load_transformers_model(model_path: Path):
    with optional_kernels_disabled():
        from transformers import AutoModel, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.eval()
        return tokenizer, model, torch


def _load_onnx_model(model_path: Path, onnx_path: Path):
    from tokenizers import Tokenizer
    import onnxruntime as ort

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return tokenizer, session


@contextmanager
def optional_kernels_disabled():
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kernels" or name.startswith("kernels."):
            raise ImportError("disabled optional kernels for local transformers import")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = real_import
