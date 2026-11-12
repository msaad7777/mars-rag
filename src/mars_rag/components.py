"""Small LangChain adapters around free, local models.

Written against langchain-core interfaces instead of langchain-community, which is
being sunset: FastEmbed embeddings, a BM25 retriever and a cross-encoder re-ranker.
"""
from __future__ import annotations

import re
from typing import Sequence

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with what which how why when who does do".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class FastEmbedEmbeddings(Embeddings):
    """bge-small-en-v1.5 through ONNX (no torch, runs on CPU)."""

    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


class BM25Retriever(BaseRetriever):
    """Keyword retriever: catches exact terms (MAV, ISRU, part numbers) that embeddings blur."""

    docs: list[Document]
    k: int = 30
    _bm25: BM25Okapi | None = None

    def model_post_init(self, __context) -> None:
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.docs[i] for i in top if scores[i] > 0]


class CrossEncoderReranker:
    """Re-scores (query, passage) pairs jointly; far sharper than bi-encoder similarity."""

    def __init__(self, model_name: str):
        self.model = TextCrossEncoder(model_name)

    def rerank(self, query: str, docs: Sequence[Document], top_n: int) -> list[Document]:
        if not docs:
            return []
        scores = list(self.model.rerank(query, [d.page_content for d in docs]))
        ranked = sorted(zip(docs, scores), key=lambda p: p[1], reverse=True)[:top_n]
        out = []
        for d, s in ranked:
            out.append(Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(s)}))
        return out
