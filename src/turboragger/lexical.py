from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

from rank_bm25 import BM25Okapi

from turboragger.contracts import RetrievalResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        source: str = "bm25",
        field: str = "all",
    ):
        if field not in {"all", "title", "text"}:
            raise ValueError(f"Unsupported BM25 field: {field}")
        self.source = source
        self.field = field
        self.doc_ids = list(corpus.keys())
        self.documents = [_document_text(doc, field=field) for doc in corpus.values()]
        self.tokenized_documents = [tokenize(document) for document in self.documents]
        self.document_token_sets = [set(tokens) for tokens in self.tokenized_documents]
        self.index = BM25Okapi(self.tokenized_documents)

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        query_token_set = set(query_tokens)
        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (
                -float(scores[index]),
                -len(query_token_set & self.document_token_sets[index]),
                self.doc_ids[index],
            ),
        )[:k]
        return [
            RetrievalResult(doc_id=self.doc_ids[index], score=float(scores[index]), source=self.source)
            for index in ranked_indices
        ]


class BM25PrfRetriever(BM25Retriever):
    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, str]],
        source: str = "bm25_prf",
        feedback_docs: int = 10,
        expansion_terms: int = 20,
        expansion_repetitions: int = 1,
    ):
        super().__init__(corpus, source=source)
        self.feedback_docs = feedback_docs
        self.expansion_terms = expansion_terms
        self.expansion_repetitions = expansion_repetitions

    def expanded_query(self, query: str) -> str:
        query_tokens = tokenize(query)
        initial = super().retrieve(query, k=self.feedback_docs)
        doc_id_to_index = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        feedback_token_lists = [
            self.tokenized_documents[doc_id_to_index[result.doc_id]]
            for result in initial
            if result.doc_id in doc_id_to_index
        ]
        expansion = select_expansion_terms(
            query_tokens=query_tokens,
            feedback_token_lists=feedback_token_lists,
            max_terms=self.expansion_terms,
        )
        repeated_terms = []
        for term in expansion:
            repeated_terms.extend([term] * self.expansion_repetitions)
        return " ".join([query, *repeated_terms]).strip()

    def retrieve(self, query: str, k: int = 100) -> list[RetrievalResult]:
        expanded = self.expanded_query(query)
        return super().retrieve(expanded, k=k)


def select_expansion_terms(
    *,
    query_tokens: list[str],
    feedback_token_lists: list[list[str]],
    max_terms: int,
) -> list[str]:
    query_token_set = set(query_tokens)
    counts: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    for tokens in feedback_token_lists:
        filtered = [
            token
            for token in tokens
            if token not in query_token_set and len(token) > 2 and not token.isdigit()
        ]
        counts.update(filtered)
        document_frequency.update(set(filtered))

    candidates = list(counts)
    candidates.sort(key=lambda token: (-document_frequency[token], -counts[token], token))
    return candidates[:max_terms]


def _document_text(doc: Mapping[str, str], field: str) -> str:
    if field == "title":
        return doc.get("title", "")
    if field == "text":
        return doc.get("text", "")
    return f"{doc.get('title', '')}\n{doc.get('text', '')}"
