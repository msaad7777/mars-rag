"""Hybrid retrieval -> cross-encoder re-rank -> grounded, cited answer, or a decline."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from . import config
from .components import BM25Retriever, CrossEncoderReranker, FastEmbedEmbeddings
from .ingest import COLLECTION

DECLINE_TOKEN = "INSUFFICIENT_CONTEXT"

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research assistant for Mars mission planning. Answer ONLY from the numbered "
     "NASA report excerpts below. Every sentence that states a fact must end with the number of "
     "the excerpt that supports it, like [2]. Do not use outside knowledge. If the excerpts do "
     "not contain the answer, reply with exactly " + DECLINE_TOKEN + " and nothing else. "
     "Be concise: at most 5 sentences."),
    ("human", "Excerpts:\n{context}\n\nQuestion: {question}"),
])


@dataclass
class Answer:
    question: str
    answer: str
    declined: bool
    decline_reason: str | None
    citations: list[dict] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_chunks() -> list[Document]:
    docs = []
    with config.CHUNKS.open() as f:
        for line in f:
            row = json.loads(line)
            text = row.pop("text")
            docs.append(Document(page_content=text, metadata=row))
    return docs


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] {d.metadata['title']} ({d.metadata['year']}), p.{d.metadata['page']}\n{d.page_content}"
        for i, d in enumerate(docs, 1))


def parse_citations(text: str, n_sources: int) -> tuple[set[int], set[int]]:
    """Return (valid, invalid) citation numbers found in the answer."""
    found = {int(n) for grp in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d+", grp)}
    valid = {n for n in found if 1 <= n <= n_sources}
    return valid, found - valid


class RAG:
    def __init__(self, llm=None, mode: str = "hybrid_rerank"):
        self.mode = mode
        self.chunks = load_chunks()
        self.embeddings = FastEmbedEmbeddings(config.EMBED_MODEL)
        store = Chroma(collection_name=COLLECTION, embedding_function=self.embeddings,
                       persist_directory=str(config.CHROMA_DIR))
        self.dense = store.as_retriever(search_kwargs={"k": config.CANDIDATES})
        self.sparse = BM25Retriever(docs=self.chunks, k=config.CANDIDATES)
        self.hybrid = EnsembleRetriever(retrievers=[self.sparse, self.dense], weights=[0.5, 0.5])
        self.reranker = CrossEncoderReranker(config.RERANK_MODEL)
        self.llm = llm or ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL,
                                     reasoning="low", temperature=0, num_ctx=8192)
        self.chain = PROMPT | self.llm

    def retrieve(self, question: str, mode: str | None = None, k: int = config.TOP_K) -> list[Document]:
        mode = mode or self.mode
        if mode == "dense":
            return self.dense.invoke(question)[:k]
        if mode == "bm25":
            return self.sparse.invoke(question)[:k]
        if mode == "bm25_rerank":
            return self.reranker.rerank(question, self.sparse.invoke(question), top_n=k)
        fused = self.hybrid.invoke(question)
        if mode == "hybrid":
            return fused[:k]
        return self.reranker.rerank(question, fused[: config.CANDIDATES * 2], top_n=k)

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        docs = self.retrieve(question)
        t1 = time.perf_counter()
        contexts = [{"n": i, "text": d.page_content, **d.metadata} for i, d in enumerate(docs, 1)]
        timings = {"retrieve": round((t1 - t0) * 1000)}

        # Gate 1: nothing relevant retrieved -> decline without spending an LLM call.
        top = docs[0].metadata.get("rerank_score", 0.0) if docs else float("-inf")
        if not docs or top < config.DECLINE_THRESHOLD:
            return Answer(question, "The NASA reports in this corpus do not cover that question.",
                          True, f"retrieval_gate (top rerank score {top:.2f})", [], contexts, timings)

        raw = self.chain.invoke({"context": format_context(docs), "question": question}).content.strip()
        timings["generate"] = round((time.perf_counter() - t1) * 1000)

        # Gate 2: the model itself says the excerpts are insufficient.
        if DECLINE_TOKEN in raw:
            return Answer(question, "The NASA reports in this corpus do not answer that question.",
                          True, "llm_insufficient_context", [], contexts, timings)

        # Gate 3: an answer with no valid citation is not grounded -> decline.
        valid, invalid = parse_citations(raw, len(docs))
        if not valid:
            return Answer(question, "I could not produce an answer grounded in the reports.",
                          True, "no_valid_citation", [], contexts, timings)
        answer = re.sub(r"\[(\d+)\]", lambda m: "" if int(m.group(1)) in invalid else m.group(0), raw)
        citations = [{"n": n, "doc_id": docs[n - 1].metadata["doc_id"], "title": docs[n - 1].metadata["title"],
                      "page": docs[n - 1].metadata["page"], "url": docs[n - 1].metadata["url"]}
                     for n in sorted(valid)]
        timings["total"] = round((time.perf_counter() - t0) * 1000)
        return Answer(question, answer, False, None, citations, contexts, timings)
