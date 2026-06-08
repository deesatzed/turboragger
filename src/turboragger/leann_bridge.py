from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping

from turboragger.contracts import RetrievalResult
from turboragger.dense import optional_kernels_disabled, require_cached_minilm_snapshot


class LeannMiniLMRetriever:
    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        model_path: Path | None = None,
        index_path: Path | None = None,
        source: str = "leann_minilm_no_recompute",
        builder_factory: Callable | None = None,
        searcher_factory: Callable | None = None,
    ):
        self.model_path = model_path or require_cached_minilm_snapshot()
        self.index_path = index_path or Path("artifacts/leann_indexes/leann_minilm_no_recompute")
        self.source = source
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_leann_home()

        self.builder_factory = builder_factory or self._default_builder_factory
        self.searcher_factory = searcher_factory or self._default_searcher_factory
        self._build_index(corpus)
        self.searcher = self.searcher_factory(str(self.index_path), recompute_embeddings=False, use_daemon=False)

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        with optional_kernels_disabled():
            results = self.searcher.search(query, top_k=k)
        converted = []
        for result in results:
            doc_id = str(getattr(result, "metadata", {}).get("node_id", getattr(result, "id", "")))
            if not doc_id:
                continue
            converted.append(RetrievalResult(doc_id=doc_id, score=float(result.score), source=self.source))
        return converted

    def _build_index(self, corpus: Mapping[str, Mapping[str, str]]) -> None:
        with optional_kernels_disabled():
            builder = self.builder_factory(
                backend_name="hnsw",
                embedding_model=str(self.model_path),
                embedding_mode="sentence-transformers",
                is_recompute=False,
                is_compact=False,
            )
            for doc_id, doc in corpus.items():
                text = f"{doc.get('title', '')}\n\n{doc.get('text', '')}"
                builder.add_text(text, metadata={"node_id": doc_id})
            builder.build_index(str(self.index_path))

    def _default_builder_factory(self, **kwargs):
        from leann.api import LeannBuilder

        return LeannBuilder(**kwargs)

    def _default_searcher_factory(self, *args, **kwargs):
        from leann.api import LeannSearcher

        return LeannSearcher(*args, **kwargs)

    @staticmethod
    def _ensure_leann_home() -> None:
        if "HOME" not in os.environ or os.environ["HOME"] == str(Path.home()):
            home = Path("artifacts/leann_runtime_home").resolve()
            home.mkdir(parents=True, exist_ok=True)
            os.environ["HOME"] = str(home)
